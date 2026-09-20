// ge_builder -- generic GE graph-building FFI (thin C ABI, C-route stage C1).
//
// The model DEFINITION lives in Rust (which ops, which descs/attrs, how they
// wire); this layer only manipulates GE C++ objects on demand. Unlike the
// raw GE operator API (Update*/SetAttr/SetInput return chainable Operator&,
// never a status), every entry here returns a checkable status -- name
// lookups, dtype parses and option requirements fail loudly instead of
// silently producing a broken model.
//
// Session model: geb_init/geb_fini bracket everything; geb_model_begin opens
// a construction slot (returns handle, becomes "current"); all add_*/set_*/
// link/graph_inputs/graph_outputs/set_option calls apply to the current
// slot; geb_model_build materializes (SetInputs/SetOutputs), compiles
// (aclgrphBuildModel) and loads (aclmdlLoadFromMem) it. Persisted OMs round-
// trip via geb_model_save / geb_model_load. Several models may coexist
// (C2 needs three OMs: vision+embed / prefix / flow).
//
// Carry-over invariants from the ge_poc battle (see skill ge-offline-om):
//   * operators must come from OperatorFactory (bare Operator is IR-less);
//   * pure operator flow only -- no AddNodeByOp (mutually exclusive with
//     SetInputs, which materializes the graph from the impl-side links);
//   * Data needs the phantom UpdateInputDesc(0U, ...) (model IO dtype
//     source) + SetAttr("index") (input slot order);
//   * build_options MUST carry input_shape (+input_format) or the compiler
//     emits a kernel-less ~10KB empty model -- enforced here, not optional;
//   * API checkability (CANN 9.0.1 operator.h): SetInput/SetAttr return
//     chainable Operator& (uncheckable), but UpdateInputDesc/UpdateOutputDesc
//     DO return graphStatus -- checked here; the post-materialization dump
//     (GEB_DUMP_GRAPH=0 to disable) remains ground truth for what linked;
#include "graph/graph.h"
#include "graph/operator_factory.h"
#include "ge/ge_ir_build.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>

extern "C" char **environ;
#include <string>
#include <vector>

using namespace ge;

namespace {

struct GebModel {
    explicit GebModel(const std::string &n) : graph(n) {}

    ge::Graph graph;
    std::map<std::string, ge::Operator> ops;
    std::vector<ge::Operator> in_ops;   // graph inputs, in binding order
    std::vector<ge::Operator> out_ops;  // graph outputs (default port)
    // outputs with explicit port index -- multi-output ops (RmsNorm's
    // REQUIRED rstd, ApplyRotaryPosEmb's q/k pair, Split) need the indexed
    // SetOutputs variant; a required output left dead-ended as an
    // intermediate node silently kills aclgrphBuildModel (empirical,
    // FormatAndShapeProcess stops without an error line)
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> out_pairs;
    std::map<ge::AscendString, ge::AscendString> options;
    ge::ModelBufferData buf{};
    uint32_t model_id = 0;
    bool loaded = false;
    // IO sizes cached at load time -- dataset assembly per run must not pay
    // an aclmdlCreateDesc/GetDesc round trip (the zero-tax runtime premise)
    std::vector<size_t> in_sizes;
    std::vector<size_t> out_sizes;
};

std::vector<GebModel> g_models;
int64_t g_cur = -1;  // construction target: last geb_model_begin

GebModel *Cur() {
    if ((g_cur < 0) || (g_cur >= static_cast<int64_t>(g_models.size()))) {
        return nullptr;
    }
    return &g_models[static_cast<size_t>(g_cur)];
}

GebModel *ByHandle(int64_t h) {
    if ((h < 0) || (h >= static_cast<int64_t>(g_models.size()))) {
        return nullptr;
    }
    return &g_models[static_cast<size_t>(h)];
}

ge::Operator *FindOp(GebModel &m, const char *name, const char *who) {
    auto it = m.ops.find(name);
    if (it == m.ops.end()) {
        std::cerr << "[geb] " << who << ": unknown operator '" << name << "'" << std::endl;
        return nullptr;
    }
    return &it->second;
}

bool ParseDtype(const char *s, ge::DataType &dt) {
    const std::string v = (s != nullptr) ? s : "";
    if ((v == "fp16") || (v == "f16")) {
        dt = ge::DT_FLOAT16;
    } else if ((v == "fp32") || (v == "f32")) {
        dt = ge::DT_FLOAT;
    } else if (v == "int32") {
        dt = ge::DT_INT32;
    } else if (v == "int64") {
        dt = ge::DT_INT64;
    } else if (v == "int8") {
        dt = ge::DT_INT8;
    } else if (v == "bool") {
        dt = ge::DT_BOOL;
    } else {
        std::cerr << "[geb] unknown dtype '" << v << "' (fp16/fp32/int32/int64/int8/bool)" << std::endl;
        return false;
    }
    return true;
}

// desc with origin shape/format set (official builders set both; infershape
// works in origin space -- see ge_poc Fp16Desc)
bool MkDesc(const int64_t *dims, int32_t n, const char *dtype, ge::TensorDesc &td) {
    if ((dims == nullptr) || (n <= 0)) {
        return false;
    }
    ge::DataType dt;
    if (!ParseDtype(dtype, dt)) {
        return false;
    }
    ge::Shape shape{std::vector<int64_t>(dims, dims + n)};  // braces: dims ctor, not a fn decl
    td = ge::TensorDesc(shape, ge::FORMAT_ND, dt);
    (void)td.SetOriginShape(shape);
    (void)td.SetOriginFormat(ge::FORMAT_ND);
    return true;
}

// post-materialization dump: node name/type, first output desc, per-input
// in-edge source+port and desc dims -- ground truth for what infershape sees
void DumpGraph(const ge::Graph &g) {
    for (const auto &gn : g.GetAllNodes()) {
        ge::AscendString gname;
        ge::AscendString gtype;
        (void)gn.GetName(gname);
        (void)gn.GetType(gtype);
        std::cerr << "[geb] node " << gname.GetString() << "(" << gtype.GetString() << ")";
        {
            ge::TensorDesc otd;
            if (gn.GetOutputDesc(0, otd) == ge::GRAPH_SUCCESS) {
                const ge::Shape osh = otd.GetShape();
                std::cerr << " out0 dims[";
                for (size_t d = 0; d < osh.GetDimNum(); d++) {
                    std::cerr << (d == 0U ? "" : ",") << osh.GetDim(d);
                }
                std::cerr << "]";
            }
        }
        for (int32_t in_idx = 0; in_idx < 8; in_idx++) {
            ge::TensorDesc td;
            if (gn.GetInputDesc(in_idx, td) != ge::GRAPH_SUCCESS) {
                continue;  // absent port (ops have sparse optional inputs)
            }
            std::string src = "<none>";
            auto peer = gn.GetInDataNodesAndPortIndexs(in_idx);
            if (peer.first != nullptr) {
                ge::AscendString pname;
                (void)peer.first->GetName(pname);
                src = std::string(pname.GetString()) + ":" + std::to_string(peer.second);
            }
            const ge::Shape sh = td.GetShape();
            std::cerr << " in" << in_idx << "<-" << src << " dims[";
            for (size_t d = 0; d < sh.GetDimNum(); d++) {
                std::cerr << (d == 0U ? "" : ",") << sh.GetDim(d);
            }
            std::cerr << "]";
        }
        std::cerr << std::endl;
    }
}

// cache IO sizes from the loaded model + print the full IO picture
int CacheIo(GebModel &m) {
    aclmdlDesc *d = aclmdlCreateDesc();
    if (d == nullptr) {
        return -100;
    }
    if (aclmdlGetDesc(d, m.model_id) != ACL_SUCCESS) {
        aclmdlDestroyDesc(d);
        return -101;
    }
    m.in_sizes.clear();
    m.out_sizes.clear();
    std::cerr << "[geb] io: n_in=" << aclmdlGetNumInputs(d) << " n_out=" << aclmdlGetNumOutputs(d)
              << std::endl;
    for (size_t i = 0; i < aclmdlGetNumInputs(d); i++) {
        m.in_sizes.push_back(aclmdlGetInputSizeByIndex(d, i));
        aclmdlIODims dims{};
        (void)aclmdlGetInputDims(d, i, &dims);
        std::cerr << "[geb]   in[" << i << "] size=" << m.in_sizes.back() << " dims=";
        for (size_t k = 0; k < dims.dimCount; k++) {
            std::cerr << (k == 0U ? "" : ",") << dims.dims[k];
        }
        std::cerr << std::endl;
    }
    for (size_t i = 0; i < aclmdlGetNumOutputs(d); i++) {
        m.out_sizes.push_back(aclmdlGetOutputSizeByIndex(d, i));
        aclmdlIODims dims{};
        (void)aclmdlGetOutputDims(d, i, &dims);
        std::cerr << "[geb]   out[" << i << "] size=" << m.out_sizes.back() << " dims=";
        for (size_t k = 0; k < dims.dimCount; k++) {
            std::cerr << (k == 0U ? "" : ",") << dims.dims[k];
        }
        std::cerr << std::endl;
    }
    aclmdlDestroyDesc(d);
    return 0;
}

}  // namespace

// ---- session ----

extern "C" int geb_init(const char *soc_version) {
    std::map<ge::AscendString, ge::AscendString> opts;
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version));
    // GEB_INIT_OPT_<key>=<value>: init 级选项直通（aclgrphBuildInitialize
    // 白名单，如 ge.enableSingleStream / ge.streamMaxParallelNum——都是
    // init scope，不在 graph build options 里）
    for (int i = 0; environ[i] != nullptr; ++i) {
        const std::string entry(environ[i]);
        const std::string prefix = "GEB_INIT_OPT_";
        const auto pos = entry.find('=');
        if (entry.rfind(prefix, 0) != 0 || pos == std::string::npos) {
            continue;
        }
        const std::string key = entry.substr(prefix.size(), pos - prefix.size());
        const std::string value = entry.substr(pos + 1);
        opts.emplace(ge::AscendString(key.c_str()), ge::AscendString(value.c_str()));
        std::cerr << "[geb] init opt: " << key << " = " << value << std::endl;
    }
    if (ge::aclgrphBuildInitialize(opts) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] aclgrphBuildInitialize failed (env checklist: python3-config on PATH, "
                     "PYTHONPATH=CANN site-packages)" << std::endl;
        return -1;
    }
    return 0;
}

extern "C" int geb_fini() {
    for (auto &m : g_models) {
        if (m.loaded) {
            (void)aclmdlUnload(m.model_id);
            m.loaded = false;
        }
    }
    g_models.clear();
    g_cur = -1;
    (void)ge::aclgrphBuildFinalize();
    return 0;
}

// ---- model lifecycle ----

extern "C" int64_t geb_model_begin(const char *graph_name) {
    g_models.emplace_back(graph_name);
    g_cur = static_cast<int64_t>(g_models.size()) - 1;
    return g_cur;
}

extern "C" int geb_model_build(int64_t h) {
    GebModel *m = ByHandle(h);
    if (m == nullptr) {
        return -1;
    }
    if (m->in_ops.empty()) {
        std::cerr << "[geb] build: no graph inputs registered (geb_graph_inputs)" << std::endl;
        return -2;
    }
    // SetInputs materializes the compute graph from the operator-level links
    // (GraphBuilderImpl::BuildGraph) -- calling it twice re-inits and fails;
    // a slot is single-build by design. NB: SetInputs/SetOutputs return a
    // chainable Graph& (not graphStatus, 9.0.1) -- IsValid below is the check
    (void)m->graph.SetInputs(m->in_ops);
    if (!m->out_pairs.empty()) {
        (void)m->graph.SetOutputs(m->out_pairs);
    } else if (!m->out_ops.empty()) {
        (void)m->graph.SetOutputs(m->out_ops);
    }
    if (!m->graph.IsValid()) {
        std::cerr << "[geb] graph invalid after SetInputs/SetOutputs (link graph before build; "
                     "AddNodeByOp is banned)" << std::endl;
        return -3;
    }
    const char *dump = getenv("GEB_DUMP_GRAPH");
    if ((dump == nullptr) || (std::string(dump) != "0")) {
        DumpGraph(m->graph);
    }
    // input_shape is mandatory: without it aclgrphBuildModel reports SUCCESS
    // but emits a kernel-less ~10KB empty model with empty IO dims (trap #3)
    if (m->options.find(ge::AscendString("input_shape")) == m->options.end()) {
        std::cerr << "[geb] build: option 'input_shape' missing" << std::endl;
        return -6;
    }
    if (m->loaded) {
        (void)aclmdlUnload(m->model_id);
        m->loaded = false;
    }
    if (ge::aclgrphBuildModel(m->graph, m->options, m->buf) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] aclgrphBuildModel failed" << std::endl;
        return -7;
    }
    std::cerr << "[geb] model built: " << m->buf.length << " bytes" << std::endl;
    if (aclmdlLoadFromMem(m->buf.data.get(), m->buf.length, &m->model_id) != ACL_SUCCESS) {
        std::cerr << "[geb] aclmdlLoadFromMem failed" << std::endl;
        return -8;
    }
    m->loaded = true;
    return CacheIo(*m);
}

extern "C" int geb_model_unload(int64_t h) {
    GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded) {
        return -1;
    }
    const aclError e = aclmdlUnload(m->model_id);
    m->loaded = false;
    return (e == ACL_SUCCESS) ? 0 : -2;
}

extern "C" int geb_model_save(int64_t h, const char *path) {
    GebModel *m = ByHandle(h);
    if ((m == nullptr) || (m->buf.data == nullptr) || (m->buf.length == 0U)) {
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return -2;
    }
    const size_t w = fwrite(m->buf.data.get(), 1U, m->buf.length, f);
    (void)fclose(f);
    return (w == m->buf.length) ? 0 : -3;
}

// load a persisted OM (no GE compile session needed -- pure ACL runtime)
extern "C" int64_t geb_model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        std::cerr << "[geb] load: cannot open " << path << std::endl;
        return -1;
    }
    (void)fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    (void)fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    if (fread(bytes.data(), 1U, bytes.size(), f) != bytes.size()) {
        (void)fclose(f);
        return -2;
    }
    (void)fclose(f);

    g_models.emplace_back("loaded");
    GebModel &m = g_models.back();
    uint32_t id = 0;
    if (aclmdlLoadFromMem(bytes.data(), bytes.size(), &id) != ACL_SUCCESS) {
        std::cerr << "[geb] load: aclmdlLoadFromMem failed" << std::endl;
        g_models.pop_back();
        return -3;
    }
    m.model_id = id;
    m.loaded = true;
    const int64_t h = static_cast<int64_t>(g_models.size()) - 1;
    // keep construction cursor where it was: loading never disturbs building
    if (CacheIo(m) != 0) {
        g_models.pop_back();
        return -4;
    }
    return h;
}

// ---- construction (current model) ----

extern "C" int geb_set_option(const char *key, const char *value) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    m->options.emplace(ge::AscendString(key), ge::AscendString(value));
    return 0;
}

extern "C" int geb_add_data(const char *name, int64_t index, const int64_t *dims, int32_t n_dims,
                            const char *dtype) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -2;
    }
    ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Data");
    // phantom INPUT desc 0: GE reads the model IO dtype/shape from
    // GetInputDescPtr(0); leaving it unset once produced an fp32-typed model
    // with an inserted Cast and all-zero output (trap #4)
    if (op.UpdateInputDesc(0U, td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] add_data '" << name << "': phantom input desc 0 rejected" << std::endl;
        return -3;
    }
    (void)op.UpdateOutputDesc("y", td);
    (void)op.SetAttr(std::string("index"), index);
    m->ops.emplace(name, op);
    return 0;
}

static bool ParseFormat(const char *s, ge::Format &fmt) {
    const std::string v = (s != nullptr) ? s : "";
    if (v == "ND") {
        fmt = ge::FORMAT_ND;
    } else if (v == "FRACTAL_NZ") {
        fmt = ge::FORMAT_FRACTAL_NZ;
    } else {
        std::cerr << "[geb] unknown format '" << v << "' (ND/FRACTAL_NZ)" << std::endl;
        return false;
    }
    return true;
}

// Data 输入（desc format 可指定）。FRACTAL_NZ 权重直入实验：MatMulV2 的
// b 算子在 310P 走 NZ，Data ND 权重会被 GE 插每执行一次的设备侧
// TransData（ND→NZ，~63GB/s 实测）——desc 直接声明 NZ 则可能免插。
extern "C" int geb_add_data_fmt(const char *name, int64_t index, const int64_t *dims, int32_t n_dims,
                                const char *dtype, const char *fmt) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::DataType dt;
    if (!ParseDtype(dtype, dt)) {
        return -2;
    }
    ge::Format f;
    if (!ParseFormat(fmt, f)) {
        return -3;
    }
    ge::Shape shape{std::vector<int64_t>(dims, dims + n_dims)};
    ge::TensorDesc td(shape, f, dt);
    (void)td.SetOriginShape(shape);
    (void)td.SetOriginFormat(f);
    ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Data");
    if (op.UpdateInputDesc(0U, td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] add_data_fmt '" << name << "': phantom input desc 0 rejected" << std::endl;
        return -4;
    }
    (void)op.UpdateOutputDesc("y", td);
    (void)op.SetAttr(std::string("index"), index);
    m->ops.emplace(name, op);
    return 0;
}

extern "C" int geb_add_op(const char *name, const char *type) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator op = ge::OperatorFactory::CreateOperator(name, type);
    m->ops.emplace(name, op);
    return 0;
}

// Const 节点（int32 一维张量 attr）——给 Reshape/LayerNormV4 的 shape 类
// 输入用。Data 输入的 shape 张量会让消费算子输出 desc 变 unknown →
// DynamicShapePartitioner 把图按未知 shape 拆子图、unknown 部分走 host
// 调度（每边界 ~20ms 停顿，vision_ma profile 取证 3220 个 unknown 标记）。
// Const 在编译期被常量折叠 → 消费算子静态 infershape。
extern "C" int geb_add_const_i32(const char *name, const int32_t *vals, int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::TensorDesc td(ge::Shape(std::vector<int64_t>(1, static_cast<int64_t>(n))), ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor t(td, reinterpret_cast<const uint8_t *>(vals), static_cast<size_t>(n) * sizeof(int32_t));
    ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Const");
    (void)op.SetAttr(std::string("value"), t);
    (void)op.UpdateOutputDesc("y", td);
    m->ops.emplace(name, op);
    return 0;
}

// Const 节点（任意 dtype/shape 的原始字节）——权重入图实验：ND Data 权重
// 每执行触发设备侧 ND→NZ TransData；Const 权重若被编译期折叠转换，则
// OM 自带 NZ 权重、零运行时税（TorchAir 378ms 的逃税路径同款假设）。
extern "C" int geb_add_const_raw(const char *name, const int64_t *dims, int32_t n_dims,
                                 const char *dtype, const uint8_t *data, int64_t len) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -2;
    }
    ge::Tensor t(td, data, static_cast<size_t>(len));
    ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Const");
    (void)op.SetAttr(std::string("value"), t);
    (void)op.UpdateOutputDesc("y", td);
    m->ops.emplace(name, op);
    return 0;
}

// DYNAMIC_INPUT ports are NOT pre-created by CreateOperatorByName (probe:
// GetDynamicInputNum("x") == 0) and the toolkit ships no op_desc.h /
// op_desc_utils.h -- but the symbols live in libgraph_base with the
// pre-CXX11-string ABI we already compile against (_GLIBCXX_USE_CXX11_ABI=0).
// Redeclare the exact signatures (from the open GE source tree) and link.
namespace ge {
class OpDesc;
class OpDescUtils {
public:
    static std::shared_ptr<OpDesc> GetOpDescFromOperator(const Operator &oprt);
};
class OpDesc {
public:
    graphStatus AddDynamicInputDesc(const std::string &name, const uint32_t num,
                                    const bool is_push_back);
};
}  // namespace ge

// Register n dynamic input ports (base name "x" -> x0..x{n-1}) on an op.
// Call right after geb_add_op, BEFORE any desc/link on those ports.
extern "C" int geb_dyn_inputs(const char *op_name, const char *base_name, int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "dyn_inputs");
    if (op == nullptr) {
        return -2;
    }
    std::shared_ptr<ge::OpDesc> desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    if (desc == nullptr) {
        return -3;
    }
    if (desc->AddDynamicInputDesc(std::string(base_name), static_cast<uint32_t>(n), true) !=
        ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] dyn_inputs: op '" << op_name << "' AddDynamicInputDesc('" << base_name
                  << "', " << n << ") failed" << std::endl;
        return -4;
    }
    return 0;
}

// Dynamic-input port count probe (diagnostics; 0 until geb_dyn_inputs runs).
extern "C" int geb_dyn_probe(const char *op_name, const char *base_name) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "dyn_probe");
    if (op == nullptr) {
        return -2;
    }
    int32_t n = op->GetDynamicInputNum(std::string(base_name));
    std::cerr << "[geb] dyn_probe: op '" << op_name << "' base '" << base_name << "' num=" << n
              << std::endl;
    return n;
}

// SetInput with a numeric dst port (DYNAMIC_INPUT ports after registration
// also work by name x0/x1, but the index form is handy for callers):
// SetInput(dst_index, src_oprt, src_index).
extern "C" int geb_link_idx(const char *dst_op, int32_t dst_index, const char *src_op,
                            int32_t src_index) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *dst = FindOp(*m, dst_op, "link_idx(dst)");
    if (dst == nullptr) {
        return -2;
    }
    if (std::string(dst_op) == std::string(src_op)) {
        (void)dst->SetInput(static_cast<uint32_t>(dst_index), *dst, static_cast<uint32_t>(src_index));
        return 0;
    }
    ge::Operator *src = FindOp(*m, src_op, "link_idx(src)");
    if (src == nullptr) {
        return -3;
    }
    (void)dst->SetInput(static_cast<uint32_t>(dst_index), *src, static_cast<uint32_t>(src_index));
    return 0;
}

extern "C" int geb_set_input_desc(const char *op_name, const char *port, const int64_t *dims,
                                  int32_t n_dims, const char *dtype) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_input_desc");
    if (op == nullptr) {
        return -2;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -3;
    }
    if (op->UpdateInputDesc(std::string(port), td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] set_input_desc: op '" << op_name << "' rejected port '" << port << "'"
                  << std::endl;
        return -4;
    }
    return 0;
}

extern "C" int geb_set_input_desc_idx(const char *op_name, int32_t port, const int64_t *dims,
                                      int32_t n_dims, const char *dtype) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_input_desc_idx");
    if (op == nullptr) {
        return -2;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -3;
    }
    if (op->UpdateInputDesc(static_cast<uint32_t>(port), td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] set_input_desc_idx: op '" << op_name << "' rejected port " << port
                  << std::endl;
        return -4;
    }
    return 0;
}

// 消费算子输入 desc（format 可指定）——NZ 直入实验的 MatMulV2 x2 用：
// dims 传 NZ 4-D [k/16, n/16, 16, 16] + FRACTAL_NZ，与上游 NZ Data 一致。
extern "C" int geb_set_input_desc_fmt(const char *op_name, const char *port, const int64_t *dims,
                                      int32_t n_dims, const char *dtype, const char *fmt) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_input_desc_fmt");
    if (op == nullptr) {
        return -2;
    }
    ge::DataType dt;
    if (!ParseDtype(dtype, dt)) {
        return -3;
    }
    ge::Format f;
    if (!ParseFormat(fmt, f)) {
        return -4;
    }
    ge::Shape shape{std::vector<int64_t>(dims, dims + n_dims)};
    ge::TensorDesc td(shape, f, dt);
    (void)td.SetOriginShape(shape);
    (void)td.SetOriginFormat(f);
    if (op->UpdateInputDesc(std::string(port), td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] set_input_desc_fmt: op '" << op_name << "' rejected port '" << port
                  << "'" << std::endl;
        return -5;
    }
    return 0;
}

extern "C" int geb_set_output_desc(const char *op_name, const char *port, const int64_t *dims,
                                   int32_t n_dims, const char *dtype) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_output_desc");
    if (op == nullptr) {
        return -2;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -3;
    }
    if (op->UpdateOutputDesc(std::string(port), td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] set_output_desc: op '" << op_name << "' rejected port '" << port << "'"
                  << std::endl;
        return -4;
    }
    return 0;
}

extern "C" int geb_set_output_desc_idx(const char *op_name, int32_t out_idx, const int64_t *dims,
                                       int32_t n_dims, const char *dtype) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_output_desc_idx");
    if (op == nullptr) {
        return -2;
    }
    ge::TensorDesc td;
    if (!MkDesc(dims, n_dims, dtype, td)) {
        return -3;
    }
    if (op->UpdateOutputDesc(static_cast<uint32_t>(out_idx), td) != ge::GRAPH_SUCCESS) {
        std::cerr << "[geb] set_output_desc_idx: op '" << op_name << "' rejected output " << out_idx
                  << std::endl;
        return -4;
    }
    return 0;
}

// ---- attrs ----

extern "C" int geb_set_attr_bool(const char *op_name, const char *attr, int32_t value) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_attr_bool");
    if (op == nullptr) {
        return -2;
    }
    (void)op->SetAttr(std::string(attr), value != 0);
    return 0;
}

extern "C" int geb_set_attr_int(const char *op_name, const char *attr, int64_t value) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_attr_int");
    if (op == nullptr) {
        return -2;
    }
    (void)op->SetAttr(std::string(attr), value);
    return 0;
}

extern "C" int geb_set_attr_float(const char *op_name, const char *attr, double value) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_attr_float");
    if (op == nullptr) {
        return -2;
    }
    (void)op->SetAttr(std::string(attr), static_cast<float>(value));
    return 0;
}

extern "C" int geb_set_attr_str(const char *op_name, const char *attr, const char *value) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_attr_str");
    if (op == nullptr) {
        return -2;
    }
    (void)op->SetAttr(std::string(attr), std::string(value));
    return 0;
}

extern "C" int geb_set_attr_int_list(const char *op_name, const char *attr, const int64_t *values,
                                     int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *op = FindOp(*m, op_name, "set_attr_int_list");
    if (op == nullptr) {
        return -2;
    }
    (void)op->SetAttr(std::string(attr), std::vector<int64_t>(values, values + n));
    return 0;
}

// ---- wiring ----

// dst.SetInput(port_name, src): the link lives on OperatorImpl both sides;
// Graph::SetInputs later materializes the compute graph from these links
extern "C" int geb_link(const char *dst_op, const char *dst_port, const char *src_op) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *dst = FindOp(*m, dst_op, "link");
    ge::Operator *src = FindOp(*m, src_op, "link");
    if ((dst == nullptr) || (src == nullptr)) {
        return -2;
    }
    (void)dst->SetInput(std::string(dst_port), *src);
    return 0;
}

// link with an explicit SOURCE output port. SetInput(dst, src_op) resolves
// the src's default output -- for multi-output ops (AddRmsNorm's
// y/rstd/x_out, ARPE's q/k) that resolution silently fails and the edge
// never attaches (dump shows in0<-<none>, compile then dies at the
// unlinked consumer). Always use this form when the src has >1 output.
extern "C" int geb_link_out(const char *dst_op, const char *dst_port, const char *src_op,
                            const char *src_out_port) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    ge::Operator *dst = FindOp(*m, dst_op, "link_out");
    ge::Operator *src = FindOp(*m, src_op, "link_out");
    if ((dst == nullptr) || (src == nullptr)) {
        return -2;
    }
    (void)dst->SetInput(std::string(dst_port), *src, std::string(src_out_port));
    return 0;
}

extern "C" int geb_graph_inputs(const char *const *names, int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    m->in_ops.clear();
    for (int32_t i = 0; i < n; i++) {
        ge::Operator *op = FindOp(*m, names[i], "graph_inputs");
        if (op == nullptr) {
            m->in_ops.clear();
            return -2;
        }
        m->in_ops.push_back(*op);
    }
    return 0;
}

extern "C" int geb_graph_outputs(const char *const *names, int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    m->out_ops.clear();
    m->out_pairs.clear();
    for (int32_t i = 0; i < n; i++) {
        ge::Operator *op = FindOp(*m, names[i], "graph_outputs");
        if (op == nullptr) {
            m->out_ops.clear();
            return -2;
        }
        m->out_ops.push_back(*op);
    }
    return 0;
}

// graph outputs with explicit output-port index: entry i binds output port
// out_idxs[i] of operator names[i] (RmsNorm rstd=1, ARPE key=1, ...)
extern "C" int geb_graph_outputs_idx(const char *const *names, const int32_t *out_idxs, int32_t n) {
    GebModel *m = Cur();
    if (m == nullptr) {
        return -1;
    }
    m->out_ops.clear();
    m->out_pairs.clear();
    for (int32_t i = 0; i < n; i++) {
        ge::Operator *op = FindOp(*m, names[i], "graph_outputs_idx");
        if (op == nullptr) {
            m->out_pairs.clear();
            return -2;
        }
        m->out_pairs.emplace_back(*op, std::vector<size_t>{static_cast<size_t>(out_idxs[i])});
    }
    return 0;
}

// ---- loaded-model IO introspection ----

extern "C" int32_t geb_num_inputs(int64_t h) {
    const GebModel *m = ByHandle(h);
    return (m != nullptr) && m->loaded ? static_cast<int32_t>(m->in_sizes.size()) : -1;
}

extern "C" int32_t geb_num_outputs(int64_t h) {
    const GebModel *m = ByHandle(h);
    return (m != nullptr) && m->loaded ? static_cast<int32_t>(m->out_sizes.size()) : -1;
}

extern "C" int64_t geb_input_size(int64_t h, int32_t idx) {
    const GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded || (idx < 0) || (static_cast<size_t>(idx) >= m->in_sizes.size())) {
        return -1;
    }
    return static_cast<int64_t>(m->in_sizes[static_cast<size_t>(idx)]);
}

extern "C" int64_t geb_output_size(int64_t h, int32_t idx) {
    const GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded || (idx < 0) ||
        (static_cast<size_t>(idx) >= m->out_sizes.size())) {
        return -1;
    }
    return static_cast<int64_t>(m->out_sizes[static_cast<size_t>(idx)]);
}

// returns n_dims (or negative on error); fills out_dims[0..min(cap,n_dims))
extern "C" int32_t geb_input_dims(int64_t h, int32_t idx, int64_t *out_dims, int32_t cap) {
    const GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded) {
        return -1;
    }
    aclmdlDesc *d = aclmdlCreateDesc();
    if ((d == nullptr) || (aclmdlGetDesc(d, m->model_id) != ACL_SUCCESS)) {
        if (d != nullptr) {
            aclmdlDestroyDesc(d);
        }
        return -2;
    }
    aclmdlIODims dims{};
    const aclError e = aclmdlGetInputDims(d, static_cast<size_t>(idx), &dims);
    aclmdlDestroyDesc(d);
    if (e != ACL_SUCCESS) {
        return -3;
    }
    for (size_t i = 0; (i < dims.dimCount) && (i < static_cast<size_t>(cap)); i++) {
        out_dims[i] = static_cast<int64_t>(dims.dims[i]);
    }
    return static_cast<int32_t>(dims.dimCount);
}

extern "C" int32_t geb_output_dims(int64_t h, int32_t idx, int64_t *out_dims, int32_t cap) {
    const GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded) {
        return -1;
    }
    aclmdlDesc *d = aclmdlCreateDesc();
    if ((d == nullptr) || (aclmdlGetDesc(d, m->model_id) != ACL_SUCCESS)) {
        if (d != nullptr) {
            aclmdlDestroyDesc(d);
        }
        return -2;
    }
    aclmdlIODims dims{};
    const aclError e = aclmdlGetOutputDims(d, static_cast<size_t>(idx), &dims);
    aclmdlDestroyDesc(d);
    if (e != ACL_SUCCESS) {
        return -3;
    }
    for (size_t i = 0; (i < dims.dimCount) && (i < static_cast<size_t>(cap)); i++) {
        out_dims[i] = static_cast<int64_t>(dims.dims[i]);
    }
    return static_cast<int32_t>(dims.dimCount);
}

// ---- run ----

// stream != nullptr -> aclmdlExecuteAsync on that stream (caller syncs);
// stream == nullptr -> synchronous aclmdlExecute (debug/discrimination variant)
extern "C" int geb_run(int64_t h, void *const *inputs, int32_t n_in, void *const *outputs,
                       int32_t n_out, void *stream) {
    GebModel *m = ByHandle(h);
    if ((m == nullptr) || !m->loaded) {
        return -1;
    }
    if ((n_in < 0) || (static_cast<size_t>(n_in) != m->in_sizes.size())) {
        std::cerr << "[geb] run: n_in " << n_in << " != model " << m->in_sizes.size() << std::endl;
        return -2;
    }
    if ((n_out < 0) || (static_cast<size_t>(n_out) != m->out_sizes.size())) {
        std::cerr << "[geb] run: n_out " << n_out << " != model " << m->out_sizes.size() << std::endl;
        return -3;
    }
    aclmdlDataset *in_ds = aclmdlCreateDataset();
    for (int32_t i = 0; i < n_in; i++) {
        (void)aclmdlAddDatasetBuffer(in_ds, aclCreateDataBuffer(inputs[i], m->in_sizes[static_cast<size_t>(i)]));
    }
    aclmdlDataset *out_ds = aclmdlCreateDataset();
    for (int32_t i = 0; i < n_out; i++) {
        (void)aclmdlAddDatasetBuffer(out_ds,
                                     aclCreateDataBuffer(outputs[i], m->out_sizes[static_cast<size_t>(i)]));
    }
    const aclError e = (stream != nullptr)
        ? aclmdlExecuteAsync(m->model_id, in_ds, out_ds, static_cast<aclrtStream>(stream))
        : aclmdlExecute(m->model_id, in_ds, out_ds);
    for (int32_t i = 0; i < n_in; i++) {
        aclDestroyDataBuffer(aclmdlGetDatasetBuffer(in_ds, static_cast<size_t>(i)));
    }
    for (int32_t i = 0; i < n_out; i++) {
        aclDestroyDataBuffer(aclmdlGetDatasetBuffer(out_ds, static_cast<size_t>(i)));
    }
    aclmdlDestroyDataset(in_ds);
    aclmdlDestroyDataset(out_ds);
    if (e != ACL_SUCCESS) {
        std::cerr << "[geb] execute err=" << static_cast<int>(e) << std::endl;
        return -4;
    }
    return 0;
}
