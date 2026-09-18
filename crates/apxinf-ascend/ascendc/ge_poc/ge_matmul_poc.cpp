// GE offline-model POC: a chain of MatMulV2 pairs built as a static GE
// graph (graph API), compiled in-memory via aclgrphBuildModel, executed
// via aclmdlExecuteAsync. Measures whether the GE static-OM runtime
// dispatches matmul tasks without the ~475us/task start tax we pay under
// ACLGraph replay (matmul_entry_probe fit), which is 80% of the 680ms vs
// torch_npu-378ms gap. No torch anywhere -- GE is a native C++ library.
#include "graph/graph.h"
#include "graph/operator_factory.h"
#include "ge/ge_ir_build.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

using namespace ge;

static ge::ModelBufferData g_model{};
static uint32_t g_model_id = 0;
static bool g_ready = false;

static ge::TensorDesc Fp16Desc(std::initializer_list<int64_t> dims) {
    ge::Shape shape{std::vector<int64_t>(dims)};
    ge::TensorDesc td(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    // the official EsGraphBuilder always sets origin shape/format on descs;
    // infershape's NodeShapeTransUtils works in origin-desc space -- without
    // this the origin shape is unknown and MatMulV2InferShape reads garbage
    (void)td.SetOriginShape(shape);
    (void)td.SetOriginFormat(ge::FORMAT_ND);
    return td;
}

extern "C" int ge_poc_init(const char *soc_version) {
    std::map<ge::AscendString, ge::AscendString> opts;
    opts.emplace(ge::AscendString("ge.socVersion"), ge::AscendString(soc_version));
    auto ret = ge::aclgrphBuildInitialize(opts);
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "[ge_poc] aclgrphBuildInitialize ret=" << static_cast<int>(ret) << std::endl;
        return -1;
    }
    return 0;
}

// graph: x[m,k] -> (MatMulV2(w1[k,n]) -> MatMulV2(w2[n,k])) x pairs -> y[m,k]
// weights enter as Data inputs so the runtime binds our device buffers.
extern "C" int ge_poc_build(int32_t pairs, int32_t m, int32_t k, int32_t n) {
    ge::Graph graph("ge_matmul_poc");

    // NB: every Operator mutator here (UpdateInputDesc/UpdateOutputDesc/
    // SetAttr/SetInput) returns a chainable Operator&, NOT graphStatus --
    // none are status-checkable; the post-materialization dump below is the
    // ground truth for what actually landed on the nodes.

    // operators must come from the factory: bare Operator(name, type) is an
    // IR-less shell whose UpdateOutputDesc/SetInput all fail
    auto mk_data = [&](const std::string &name, int64_t index,
                       std::initializer_list<int64_t> dims) -> ge::Operator {
        ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Data");
        // Update*/SetAttr return chainable Operator&, not graphStatus -- their
        // effect is validated by the post-materialization dump below
        // Data carries a phantom INPUT desc 0: GE's Impl::SetInputs reads the
        // model IO dtype+shape from GetInputDescPtr(0) -- unset dtype made the
        // first model fp32 (in size 4 = one fp32) with a Cast inserted, and
        // our fp16 bits read as denormal fp32 -> all-zero output
        (void)op.UpdateInputDesc(0U, Fp16Desc(dims));
        (void)op.UpdateOutputDesc("y", Fp16Desc(dims));
        (void)op.SetAttr("index", static_cast<int64_t>(index));
        return op;
    };
    // GE_POC_OP selects the cube op: matmulv2 (default) or the classic MatMul
    // (the MatMulV2InferShape rank-2/4 gate is V2-specific; MatMul is the
    // atc-classic offline path)
    const char *op_sel = getenv("GE_POC_OP");
    const bool use_v2 = (op_sel == nullptr) || (std::string(op_sel) != "matmul");
    const char *mm_type = use_v2 ? "MatMulV2" : "MatMul";
    auto mk_mm = [&](const std::string &name, std::initializer_list<int64_t> in1,
                     std::initializer_list<int64_t> in2,
                     std::initializer_list<int64_t> out) -> ge::Operator {
        ge::Operator op = ge::OperatorFactory::CreateOperator(name, mm_type);
        (void)op.UpdateInputDesc("x1", Fp16Desc(in1));
        (void)op.UpdateInputDesc("x2", Fp16Desc(in2));
        (void)op.UpdateOutputDesc("y", Fp16Desc(out));
        if (use_v2) {
            (void)op.SetAttr("transpose_x1", false);
            (void)op.SetAttr("transpose_x2", false);
        } else {
            (void)op.SetAttr("transpose_a", false);
            (void)op.SetAttr("transpose_b", false);
        }
        return op;
    };

    ge::Operator x_op = mk_data("x", 0, {m, k});
    ge::Operator w1_op = mk_data("w1", 1, {k, n});
    ge::Operator w2_op = mk_data("w2", 2, {n, k});

    // pure-operator assembly: SetInput links live on OperatorImpl (input_link_
    // on the dst + output_links_ on the src); Graph::SetInputs materializes
    // the whole compute graph by walking those links (GraphBuilderImpl::
    // BuildGraph, operator.cc). AddNodeByOp must NOT be mixed in: it
    // SetValid()s an empty inner graph (locking SetInputs, graph.cc "Inner
    // graph has been inited") and its nodes carry no operator-level edges.
    ge::Operator cur = x_op;
    for (int32_t i = 0; i < pairs; i++) {
        ge::Operator mm1 = mk_mm(std::string("mm1_") + std::to_string(i), {m, k}, {k, n}, {m, n});
        (void)mm1.SetInput("x1", cur);
        (void)mm1.SetInput("x2", w1_op);

        ge::Operator mm2 = mk_mm(std::string("mm2_") + std::to_string(i), {m, n}, {n, k}, {m, k});
        (void)mm2.SetInput("x1", mm1);
        (void)mm2.SetInput("x2", w2_op);
        cur = mm2;
    }
    (void)graph.SetInputs({x_op, w1_op, w2_op});
    (void)graph.SetOutputs({cur});
    if (!graph.IsValid()) {
        std::cerr << "[ge_poc] SetInputs failed to materialize the graph" << std::endl;
        return -3;
    }

    // dump the materialized topology + input descs: node name/type, per-input
    // in-edge source and desc dims -- the ground truth the infershape pass sees
    for (const auto &gn : graph.GetAllNodes()) {
        ge::AscendString gname;
        ge::AscendString gtype;
        (void)gn.GetName(gname);
        (void)gn.GetType(gtype);
        std::cerr << "[ge_poc] node " << gname.GetString() << "(" << gtype.GetString() << ")";
        {
            // every node's first output desc (the graph IO comes from these)
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
        for (int32_t in_idx = 0; in_idx < 2; in_idx++) {
            ge::TensorDesc td;
            std::string src = "<none>";
            if (gn.GetInputDesc(in_idx, td) == ge::GRAPH_SUCCESS) {
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
        }
        std::cerr << std::endl;
    }

    // atc semantics: Impl::SetInputs resolves Data-node shapes/dtype from the
    // input_shape OPTION map (Data has no input desc -- GetInputDescPtr(0) on
    // the descs alone yields unknown dims and an empty 9KB kernel-less model)
    std::map<ge::AscendString, ge::AscendString> build_options;
    build_options.emplace(ge::AscendString("input_format"), ge::AscendString("ND"));
    const std::string input_shape = "x:" + std::to_string(m) + "," + std::to_string(k) +
                                    ";w1:" + std::to_string(k) + "," + std::to_string(n) +
                                    ";w2:" + std::to_string(n) + "," + std::to_string(k);
    build_options.emplace(ge::AscendString("input_shape"), ge::AscendString(input_shape.c_str()));
    auto ret = ge::aclgrphBuildModel(graph, build_options, g_model);
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "[ge_poc] aclgrphBuildModel ret=" << static_cast<int>(ret) << std::endl;
        return -4;
    }
    std::cerr << "[ge_poc] model built: " << g_model.length << " bytes" << std::endl;
    // GE_POC_SAVE=path.om dumps the buffer for offline inspection
    const char *save_path = getenv("GE_POC_SAVE");
    if (save_path != nullptr) {
        FILE *f = fopen(save_path, "wb");
        if (f != nullptr) {
            (void)fwrite(g_model.data.get(), 1U, g_model.length, f);
            (void)fclose(f);
            std::cerr << "[ge_poc] model saved to " << save_path << std::endl;
        }
    }

    aclError err = aclmdlLoadFromMem(g_model.data.get(), g_model.length, &g_model_id);
    if (err != ACL_SUCCESS) {
        std::cerr << "[ge_poc] aclmdlLoadFromMem err=" << static_cast<int>(err) << std::endl;
        return -5;
    }
    g_ready = true;
    return 0;
}

// Bind caller-owned device buffers as the model IO and execute on the
// given stream (dataset assembly per call; buffers themselves untouched).
extern "C" int ge_poc_run(void *x, void *w1, void *w2, void *y, void *stream) {
    if (!g_ready) {
        return -10;
    }
    aclmdlDesc *desc = aclmdlCreateDesc();
    aclError err = aclmdlGetDesc(desc, g_model_id);
    if (err != ACL_SUCCESS) {
        std::cerr << "[ge_poc] aclmdlGetDesc err=" << static_cast<int>(err) << std::endl;
        aclmdlDestroyDesc(desc);
        return -11;
    }
    // sizes from the compiled model (index: 0=x,1=w1,2=w2 / out 0=y)
    const size_t in_sz[3] = {aclmdlGetInputSizeByIndex(desc, 0U), aclmdlGetInputSizeByIndex(desc, 1U),
                             aclmdlGetInputSizeByIndex(desc, 2U)};
    const size_t out_sz = aclmdlGetOutputSizeByIndex(desc, 0U);
    static bool sizes_logged = false;
    if (!sizes_logged) {
        // full IO picture: counts, dims, dtypes, sizes -- what does the
        // compiled model actually think its interface is?
        std::cerr << "[ge_poc] model io: n_in=" << aclmdlGetNumInputs(desc)
                  << " n_out=" << aclmdlGetNumOutputs(desc) << std::endl;
        for (size_t i = 0; i < aclmdlGetNumInputs(desc); i++) {
            aclmdlIODims dims{};
            (void)aclmdlGetInputDims(desc, i, &dims);
            std::cerr << "[ge_poc]   in[" << i << "] size=" << aclmdlGetInputSizeByIndex(desc, i) << " dims=";
            for (size_t d = 0; d < dims.dimCount; d++) {
                std::cerr << (d == 0U ? "" : ",") << dims.dims[d];
            }
            std::cerr << std::endl;
        }
        for (size_t i = 0; i < aclmdlGetNumOutputs(desc); i++) {
            aclmdlIODims dims{};
            (void)aclmdlGetOutputDims(desc, i, &dims);
            std::cerr << "[ge_poc]   out[" << i << "] size=" << aclmdlGetOutputSizeByIndex(desc, i) << " dims=";
            for (size_t d = 0; d < dims.dimCount; d++) {
                std::cerr << (d == 0U ? "" : ",") << dims.dims[d];
            }
            std::cerr << std::endl;
        }
        sizes_logged = true;
    }
    aclmdlDestroyDesc(desc);

    void *in_ptrs[3] = {x, w1, w2};
    aclmdlDataset *in_ds = aclmdlCreateDataset();
    for (size_t i = 0; i < 3U; i++) {
        aclDataBuffer *buf = aclCreateDataBuffer(in_ptrs[i], in_sz[i]);
        aclmdlAddDatasetBuffer(in_ds, buf);
    }
    aclmdlDataset *out_ds = aclmdlCreateDataset();
    aclDataBuffer *obuf = aclCreateDataBuffer(y, out_sz);
    aclmdlAddDatasetBuffer(out_ds, obuf);

    // GE_POC_SYNC=1 -> synchronous execute (async returned success but wrote
    // nothing; discriminates stream-async semantics from a deeper model/run
    // mismatch)
    const char *sync_sel = getenv("GE_POC_SYNC");
    if ((sync_sel != nullptr) && (std::string(sync_sel) == "1")) {
        err = aclmdlExecute(g_model_id, in_ds, out_ds);
    } else {
        err = aclmdlExecuteAsync(g_model_id, in_ds, out_ds, static_cast<aclrtStream>(stream));
    }

    for (size_t i = 0; i < 3U; i++) {
        aclDestroyDataBuffer(aclmdlGetDatasetBuffer(in_ds, i));
    }
    aclDestroyDataBuffer(aclmdlGetDatasetBuffer(out_ds, 0U));
    aclmdlDestroyDataset(in_ds);
    aclmdlDestroyDataset(out_ds);
    if (err != ACL_SUCCESS) {
        std::cerr << "[ge_poc] execute err=" << static_cast<int>(err) << std::endl;
        return -12;
    }
    return 0;
}

extern "C" int ge_poc_fini() {
    if (g_model_id != 0U) {
        (void)aclmdlUnload(g_model_id);
        g_model_id = 0;
    }
    g_ready = false;
    (void)ge::aclgrphBuildFinalize();
    return 0;
}
