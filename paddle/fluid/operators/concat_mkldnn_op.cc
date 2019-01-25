/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::concat;
using mkldnn::stream;
using platform::to_void_cast;

static void EnforceLayouts(const std::vector<const Tensor*> inputs) {
  for (auto* input : inputs) {
    const bool is_layout_correct = input->layout() == DataLayout::kMKLDNN;
    const bool is_format_defined =
        input->format() != memory::format::format_undef;
    PADDLE_ENFORCE(is_layout_correct && is_format_defined,
                   "Wrong layout/format set for Input tensor");
  }
}

static memory::primitive_desc CreateMemPrimDesc(const Tensor& input,
                                                const mkldnn::engine& engine,
                                                const memory::data_type& dt) {
  const auto dims = paddle::framework::vectorize2int(input.dims());
  const auto format = input.format();
  auto description = memory::desc(dims, dt, format);
  auto mem_prim_desc = memory::primitive_desc(description, engine);
  return mem_prim_desc;
}

static mkldnn::memory::format GetDstMemFormat(
    const concat::primitive_desc& concat_pd) {
  return (memory::format)concat_pd.dst_primitive_desc().desc().data.format;
}

static platform::CPUPlace GetCpuPlace(
    const paddle::framework::ExecutionContext& ctx) {
  auto place = ctx.GetPlace();
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(place),
                 "It must use CPUPlace.");
  return boost::get<platform::CPUPlace>(place);
}

static const mkldnn::engine& GetMKLDNNEngine(
    const paddle::framework::ExecutionContext& ctx) {
  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
  return dev_ctx.GetEngine();
}

std::string CreateKey(const paddle::framework::ExecutionContext& ctx,
                      const std::vector<const Tensor*> multi_input,
                      const int64_t& concat_axis, const memory::data_type& dt) {
  std::string key;
  key.reserve(MaxKeyLength);
  for (size_t i = 0; i < multi_input.size(); i++) {
    platform::ConvMKLDNNHandler::AppendKeyDims(
        &key, paddle::framework::vectorize2int(multi_input[i]->dims()));
  }
  platform::ConvMKLDNNHandler::AppendKey(&key, std::to_string(concat_axis));
  platform::ConvMKLDNNHandler::AppendKey(&key, ctx.op().Output("Out"));
  platform::ConvMKLDNNHandler::AppendKey(&key, std::to_string(dt));
  return key;
}

template <typename T>
class ConcatPrimitiveFactory {
 public:
  concat::primitive_desc CreateConcatPrimDescriptor(
      const std::vector<const Tensor*> multi_input, Tensor* output,
      int concat_axis, const mkldnn::engine& mkldnn_engine,
      const memory::data_type& dt = memory::data_type::f32) {
    CreateSourcesDescriptors(multi_input, mkldnn_engine, dt);
    auto dst_desc = CreateDstMemDescriptor(output, dt);
    return concat::primitive_desc(dst_desc, concat_axis, srcs_pd);
  }

  concat CreateConcatPrimitive(const concat::primitive_desc& concat_pd,
                               Tensor* output, platform::CPUPlace place) {
    CreateSourcePrimitiveAts();
    dst_mem = CreateDstMemory(concat_pd, output, place);
    return concat(concat_pd, inputs, dst_mem.get());
  }

 private:
  memory::desc CreateDstMemDescriptor(Tensor* output,
                                      const memory::data_type& dt) {
    auto dst_dims = paddle::framework::vectorize2int(output->dims());
    return memory::desc(dst_dims, dt, memory::format::any);
  }

  mkldnn::memory CreateDstMemory(const concat::primitive_desc& concat_pd,
                                 Tensor* output,
                                 const platform::CPUPlace& place) {
    return memory(concat_pd.dst_primitive_desc(),
                  output->mutable_data<T>(place));
  }

  void CreateSourcesDescriptors(const std::vector<const Tensor*> multi_input,
                                const mkldnn::engine& mkldnn_engine,
                                const memory::data_type& dt) {
    for (size_t i = 0; i < multi_input.size(); i++) {
      auto mem_prim_desc =
          CreateMemPrimDesc(*multi_input[i], mkldnn_engine, dt);
      srcs_pd.push_back(mem_prim_desc);
      srcs.push_back(
          memory(mem_prim_desc, to_void_cast(multi_input[i]->data<T>())));
    }
  }

  void CreateSourcePrimitiveAts() {
    inputs.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); i++) {
      inputs.push_back(srcs[i]);
    }
  }

 private:
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<primitive::at> inputs;

 public:
  std::vector<memory> srcs;
  boost::optional<memory> dst_mem;  // TODO(mgallus): change to std::optional
};                                  // upon introduction of C++17 to paddle

template <typename T>
class ConcatMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto place = GetCpuPlace(ctx);
    const auto& mkldnn_engine = GetMKLDNNEngine(ctx);

    auto multi_input = ctx.MultiInput<Tensor>("X");
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");
    int64_t concat_axis = static_cast<int64_t>(ctx.Attr<int>("axis"));

    bool in_place = true;
    for (auto i = 0; i < multi_input.size(); i++) {
      if (concat_axis != 0 && multi_input[i]->dims()[concat_axis - 1] != 1) {
        in_place = false;
        break;
      }
    }

    ConcatPrimitiveFactory<T> prim_creator;
    auto concat_pd = prim_creator.CreateConcatPrimDescriptor(
        multi_input, output, static_cast<int>(concat_axis), mkldnn_engine);
    auto concat = prim_creator.CreateConcatPrimitive(concat_pd, output, place);

    if (in_place) {
      size_t offset = 0;
      for (auto i = 0; i < multi_input.size(); i++) {
        memcpy(static_cast<T*>(prim_creator.dst_mem.get().get_data_handle()) +
                   offset,
               static_cast<T*>(prim_creator.srcs[i].get_data_handle()),
               sizeof(T) * multi_input[i]->numel());
        prim_creator.srcs[i].set_data_handle(
            static_cast<T*>(prim_creator.dst_mem.get().get_data_handle()) +
            offset);
        offset += multi_input[i]->numel();
      }
    } else {
      stream(stream::kind::eager).submit({concat}).wait();
    }

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetDstMemFormat(concat_pd));
  }
};

template <typename T>
class ConcatINT8MKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto multi_input = ctx.MultiInput<Tensor>("X");
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");
    int64_t concat_axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    auto place = GetCpuPlace(ctx);

    memory::data_type dt =
        paddle::framework::ToMKLDNNDataType(multi_input[0]->type());

    bool in_place = true;
    for (auto i = 0; i < multi_input.size(); i++) {
      if (concat_axis != 0 && multi_input[i]->dims()[concat_axis - 1] != 1) {
        in_place = false;
        break;
      }
    }

    ConcatPrimitiveFactory<T> prim_creator;
    std::string key = CreateKey(ctx, multi_input, concat_axis, dt);
    const std::string key_prim = key + "@concat_p";
    const std::string key_concat_pd = key + "@concat_pd";

    std::shared_ptr<concat::primitive_desc> concat_pd;
    auto concat_p = std::static_pointer_cast<concat>(dev_ctx.GetBlob(key_prim));

    if (concat_p == nullptr) {
      const auto& mkldnn_engine = dev_ctx.GetEngine();
      concat_pd = std::make_shared<concat::primitive_desc>(
          prim_creator.CreateConcatPrimDescriptor(multi_input, output,
                                                  static_cast<int>(concat_axis),
                                                  mkldnn_engine, dt));
      concat_p = std::make_shared<concat>(
          prim_creator.CreateConcatPrimitive(*concat_pd, output, place));

      if (in_place) {
        size_t offset = 0;
        for (auto i = 0; i < multi_input.size(); i++) {
          memcpy(static_cast<T*>(prim_creator.dst_mem.get().get_data_handle()) +
                     offset,
                 static_cast<T*>(prim_creator.srcs[i].get_data_handle()),
                 sizeof(T) * multi_input[i]->numel());
          prim_creator.srcs[i].set_data_handle(
              static_cast<T*>(prim_creator.dst_mem.get().get_data_handle()) +
              offset);
          offset += multi_input[i]->numel();
        }
      }

      dev_ctx.SetBlob(key_prim, concat_p);
      dev_ctx.SetBlob(key_concat_pd, concat_pd);
    } else {
      concat_pd = std::static_pointer_cast<concat::primitive_desc>(
          dev_ctx.GetBlob(key_concat_pd));
      for (size_t i = 0; i < multi_input.size(); i++) {
        prim_creator.srcs[i].set_data_handle(
            to_void_cast<T>(multi_input[i]->data<T>()));
      }
      prim_creator.dst_mem.get().set_data_handle(
          output->mutable_data<T>(place));
    }

    if (!in_place) stream(stream::kind::eager).submit({*concat_p}).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetDstMemFormat(*concat_pd));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(concat, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConcatMKLDNNFP32,
                                    ops::ConcatMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(concat, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kConcatMKLDNNINT8,
                                    ops::ConcatINT8MKLDNNOpKernel<int8_t>);
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(concat, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kConcatMKLDNNINT8,
                                    ops::ConcatINT8MKLDNNOpKernel<uint8_t>);
