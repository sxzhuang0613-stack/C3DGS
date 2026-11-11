"""验证缓存错配及损坏索引被拦截，避免用真实非法访问作为测试结果。"""
import ctypes
from dataclasses import replace
import unittest

import torch
import diff_gaussian_slice_rendering as package


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 与重新构建的渲染扩展")
class RendererSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package._require_extension()

    def native_forward(self, count=2, debug=False, height=17, width=19):
        cfg = package.GaussianRasterizationSettings.from_axis(height, width, debug=debug)
        means = torch.zeros(count, 3, device="cuda")
        intensities = torch.ones(count, 1, device="cuda")
        empty = means.new_empty(0)
        covariance = means.new_tensor([[.1, 0., 0., .1, 0., .1]]).repeat(count, 1)
        args = (means, intensities, empty, empty, covariance,
                cfg.slice_to_world.cuda().float().contiguous(),
                cfg.psf_vector.cuda().float().contiguous(), height, width,
                cfg.background, cfg.kernel_cutoff, cfg.cutoff_transition,
                cfg.covariance_jitter, debug)
        return args, package._C.rasterize_gaussians(*args)

    def native_backward(self, args, result):
        return package._C.rasterize_gaussians_backward(
            *args, torch.ones_like(result[1]), result[0], *result[3:6])

    def test_mixed_forward_caches_rejected_even_with_equal_shapes(self):
        for count in (0, 2):
            args, first = self.native_forward(count)
            _, second = self.native_forward(count)
            for index in (3, 4, 5):
                mixed = list(first)
                mixed[index] = second[index]
                with self.subTest(count=count, cache=index):
                    with self.assertRaisesRegex(RuntimeError, "backward.cache_header"):
                        self.native_backward(args, mixed)
            # 检查拒绝发生在 GPU 越界之前，原始缓存及 CUDA 上下文仍可正常使用。
            self.native_backward(args, first)
            torch.cuda.synchronize()

    def test_invalid_header_and_short_cache_rejected(self):
        args, result = self.native_forward()
        for index in (3, 4, 5):
            wrong_kind = result[index].clone()
            wrong_kind[:64].view(torch.int64)[7] = 99
            for cache in (torch.zeros_like(result[index]), result[index][:1], wrong_kind):
                broken = list(result)
                broken[index] = cache
                with self.assertRaisesRegex(RuntimeError, "forward cache"):
                    self.native_backward(args, broken)
        self.native_backward(args, result)

    def test_cache_dimensions_and_instance_count_checked(self):
        args, result = self.native_forward()
        # 同像素数、同 tile 数的尺寸变更不能只靠缓存字节数发现。
        changed = list(args)
        changed[7], changed[8] = args[8], args[7]
        changed_result = list(result)
        changed_result[1] = result[1].transpose(1, 2).contiguous()
        with self.assertRaisesRegex(RuntimeError, "backward.cache_header"):
            self.native_backward(changed, changed_result)
        changed_result = list(result)
        changed_result[0] -= 1
        with self.assertRaisesRegex(RuntimeError, "backward.cache_header"):
            self.native_backward(args, changed_result)

    def test_debug_rejects_bad_range_and_gaussian_id(self):
        # 布局按 native obtain 的 128 字节对齐规则定位，头部为八个 uint64。
        for corruption, message in (("range", "tile range"), ("id", "Gaussian ID")):
            args, result = self.native_forward(debug=True)
            broken = list(result)
            if corruption == "range":
                image = result[5].clone()
                image[128:136].view(torch.int32).copy_(
                    torch.tensor([0, result[0] + 1], device="cuda", dtype=torch.int32))
                broken[5] = image
            else:
                count = result[0]
                offset = 128
                # 分别跳过两个 uint64 key 数组和一个 uint32 未排序 ID 数组。
                for size in (8 * count, 8 * count, 4 * count):
                    offset = (offset + size + 127) // 128 * 128
                binning = result[4].clone()
                binning[offset:offset + 4 * count].view(torch.int32).fill_(args[0].shape[0])
                broken[4] = binning
            with self.subTest(corruption=corruption):
                with self.assertRaisesRegex(RuntimeError, "backward.render: " + message):
                    self.native_backward(args, broken)
            self.native_backward(args, result)
            torch.cuda.synchronize()

    def test_debug_preserves_outputs_and_gradients_on_active_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for precomputed in (False, True):
                means = torch.tensor([[.1, -.2, .05]], device="cuda", requires_grad=True)
                intensity = torch.tensor([[.7]], device="cuda", requires_grad=True)
                scales = torch.full((1, 3), .3, device="cuda", requires_grad=True)
                rotations = torch.tensor([[1., .2, .1, 0.]], device="cuda", requires_grad=True)
                covariance = torch.tensor([[.1, 0., 0., .1, 0., .1]], device="cuda", requires_grad=True)
                cfg = package.GaussianRasterizationSettings.from_axis(17, 19)
                parameters = ({"cov3Ds_precomp": covariance} if precomputed else
                              {"scales": scales, "rotations": rotations})
                inputs = (means, intensity, covariance) if precomputed else (means, intensity, scales, rotations)
                values = []
                for debug in (False, True):
                    outputs = package.rasterize_gaussians(
                        means, intensity, replace(cfg, debug=debug), **parameters)
                    gradients = torch.autograd.grad(outputs[0].square().mean(), inputs)
                    values.append((outputs, gradients))
                for ordinary, debug in zip(*[v[0] for v in values]):
                    torch.testing.assert_close(ordinary, debug, rtol=0, atol=0)
                for ordinary, debug in zip(*[v[1] for v in values]):
                    torch.testing.assert_close(ordinary, debug, rtol=1e-5, atol=1e-6)
            stream.synchronize()

    def test_grid_limit_rejected_before_kernel_launch(self):
        driver = ctypes.CDLL("libcuda.so.1")
        limit = ctypes.c_int()
        self.assertEqual(driver.cuInit(0), 0)
        # CUDA Driver API 的 MAX_GRID_DIM_Y 属性值为 6，设备使用当前 PyTorch ordinal。
        device = ctypes.c_int()
        self.assertEqual(driver.cuCtxGetDevice(ctypes.byref(device)), 0)
        self.assertEqual(driver.cuDeviceGetAttribute(ctypes.byref(limit), 6, device), 0)
        with self.assertRaisesRegex(RuntimeError, "image dimensions exceed CUDA grid limits"):
            self.native_forward(count=0, height=limit.value * 16 + 1, width=1)
        self.native_forward()


if __name__ == "__main__":
    unittest.main()
