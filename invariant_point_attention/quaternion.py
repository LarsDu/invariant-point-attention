# The following code was copied by Larry Du from pytorch3d due to
# dependency issues

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file at https://github.com/facebookresearch/pytorch3d


import torch

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
            """
            Convert rotations given as quaternions to rotation matrices.

            Args:
                quaternions: quaternions with real part first,
                    as tensor of shape (..., 4).

            Returns:
                Rotation matrices as tensor of shape (..., 3, 3).
            """
            r, i, j, k = torch.unbind(quaternions, -1)
            # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
            two_s = 2.0 / (quaternions * quaternions).sum(-1)

            o = torch.stack(
                (
                    1 - two_s * (j * j + k * k),
                    two_s * (i * j - k * r),
                    two_s * (i * k + j * r),
                    two_s * (i * j + k * r),
                    1 - two_s * (i * i + k * k),
                    two_s * (j * k - i * r),
                    two_s * (i * k - j * r),
                    two_s * (j * k + i * r),
                    1 - two_s * (i * i + j * j),
                ),
                -1,
            )
            return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


