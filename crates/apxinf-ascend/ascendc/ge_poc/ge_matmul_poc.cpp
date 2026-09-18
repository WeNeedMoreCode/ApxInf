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

#include <iostream>
#include <map>
#include <string>

using namespace ge;

static ge::ModelBufferData g_model{};
static uint32_t g_model_id = 0;
static bool g_ready = false;

static ge::TensorDesc Fp16Desc(std::initializer_list<int64_t> dims) {
    return ge::TensorDesc(ge::Shape(std::vector<int64_t>(dims)), ge::FORMAT_ND, ge::DT_FLOAT16);
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

    // operators must come from the factory: bare Operator(name, type) is an
    // IR-less shell whose UpdateOutputDesc/SetInput all fail
    auto mk_data = [&](const std::string &name, int64_t index,
                       std::initializer_list<int64_t> dims) -> ge::Operator {
        ge::Operator op = ge::OperatorFactory::CreateOperator(name, "Data");
        op.UpdateOutputDesc(0U, Fp16Desc(dims));
        op.SetAttr("index", static_cast<int64_t>(index));
        return op;
    };
    auto mk_mm = [&](const std::string &name, std::initializer_list<int64_t> in1,
                     std::initializer_list<int64_t> in2,
                     std::initializer_list<int64_t> out) -> ge::Operator {
        ge::Operator op = ge::OperatorFactory::CreateOperator(name, "MatMulV2");
        op.UpdateInputDesc("x1", Fp16Desc(in1));
        op.UpdateInputDesc("x2", Fp16Desc(in2));
        op.UpdateOutputDesc("y", Fp16Desc(out));
        op.SetAttr("transpose_x1", false);
        op.SetAttr("transpose_x2", false);
        return op;
    };

    ge::Operator x_op = mk_data("x", 0, {m, k});
    ge::Operator w1_op = mk_data("w1", 1, {k, n});
    ge::Operator w2_op = mk_data("w2", 2, {n, k});

    ge::Operator cur = x_op;
    std::vector<ge::Operator> nodes{x_op, w1_op, w2_op};
    for (int32_t i = 0; i < pairs; i++) {
        ge::Operator mm1 = mk_mm(std::string("mm1_") + std::to_string(i), {m, k}, {k, n}, {m, n});
        mm1.SetInput("x1", cur);
        mm1.SetInput("x2", w1_op);
        nodes.push_back(mm1);

        ge::Operator mm2 = mk_mm(std::string("mm2_") + std::to_string(i), {m, n}, {n, k}, {m, k});
        mm2.SetInput("x1", mm1);
        mm2.SetInput("x2", w2_op);
        nodes.push_back(mm2);
        cur = mm2;
    }
    for (auto &op : nodes) {
        (void)graph.AddNodeByOp(op);
    }
    // IR build requires explicit input/output binding on the graph
    (void)graph.SetInputs({x_op, w1_op, w2_op});
    (void)graph.SetOutputs({cur});

    std::map<ge::AscendString, ge::AscendString> build_options;
    auto ret = ge::aclgrphBuildModel(graph, build_options, g_model);
    if (ret != ge::GRAPH_SUCCESS) {
        std::cerr << "[ge_poc] aclgrphBuildModel ret=" << static_cast<int>(ret) << std::endl;
        return -4;
    }
    std::cerr << "[ge_poc] model built: " << g_model.length << " bytes" << std::endl;

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

    err = aclmdlExecuteAsync(g_model_id, in_ds, out_ds, static_cast<aclrtStream>(stream));

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
