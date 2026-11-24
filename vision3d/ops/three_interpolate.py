from torch import Tensor
from torch.autograd import Function

from vision3d.utils.misc import load_ext

ext_module = load_ext("vision3d.ext_gpu", ["three_interpolate_forward", "three_interpolate_backward"])


class _ThreeInterpolateFunction(Function):
    @staticmethod
    def forward(ctx, s_feats: Tensor, indices: Tensor, weights: Tensor, transposed: bool = False) -> Tensor:
        """
        Interpolate the features for the query points from the support points.

        Three support points are used to interpolate one query point.

        Args:
            s_feats (Tensor): The features of the support points with shape (B, C, M).
            indices (Tensor): The indices of the 3-NN of the query points with shape (B, N, 3).
            weights (Tensor): The weights of the 3-NN of the query pionts with shape (B, N, 3).
            transposed (bool=False): If True, the s_feats is with shape (B, M, C).

        Returns:
            A Tensor of the interpolated features with shape (B, C, N) or (B, N, C) if transposed.
        """
        if transposed:
            s_feats = s_feats.transpose(1, 2)  # (B, M, C) -> (B, C, M)
        s_feats = s_feats.contiguous()
        indices = indices.contiguous()
        weights = weights.contiguous()

        ctx.save_for_backward(indices, weights)
        ctx.num_supports = s_feats.shape[-1]
        ctx.transposed = transposed

        q_feats = s_feats.new_zeros(size=(s_feats.shape[0], s_feats.shape[1], indices.shape[1]))  # (B, C, N)

        ext_module.three_interpolate_forward(s_feats, indices, weights, q_feats)

        if transposed:
            q_feats = q_feats.transpose(1, 2).contiguous()  # (B, C, N) -> (B, N, C)

        return q_feats

    @staticmethod
    def backward(ctx, q_grads: Tensor) -> tuple[Tensor, None, None, None]:  # type: ignore
        indices, weights = ctx.saved_tensors
        num_supports = ctx.num_supports
        transposed = ctx.transposed

        if transposed:
            q_grads = q_grads.transpose(1, 2)  # (B, N, C) ->  (B, C, N)
        q_grads = q_grads.contiguous()

        s_grads = q_grads.new_zeros(size=(q_grads.shape[0], q_grads.shape[1], num_supports))  # (B, C, M)

        ext_module.three_interpolate_backward(q_grads, indices, weights, s_grads, num_supports)

        if transposed:
            s_grads = s_grads.transpose(1, 2).contiguous()  # (B, C, M) -> (B, M, C)

        return s_grads, None, None, None


def three_interpolate(s_feats: Tensor, indices: Tensor, weights: Tensor, transposed: bool = False) -> Tensor:
    return _ThreeInterpolateFunction.apply(s_feats, indices, weights, transposed=transposed)  # type: ignore
