import torch


def get_orthogonal(normals):
    normals_x, normals_y, normals_z = torch.unbind(normals, dim=-1)
    tangent = torch.zeros_like(normals)
    tangent_x, tangent_y, tangent_z = torch.unbind(tangent, dim=-1)

    idx = torch.abs(normals_x) > torch.abs(normals_y)

    invLen = 1.0 / torch.sqrt(normals_x[idx] * normals_x[idx] + normals_z[idx] * normals_z[idx])
    tangent_x[idx] = invLen * normals_z[idx]
    tangent_z[idx] = -invLen * normals_x[idx]

    invLen = 1.0 / torch.sqrt(normals_y[~idx] * normals_y[~idx] + normals_z[~idx] * normals_z[~idx])
    tangent_y[~idx] = invLen * normals_z[~idx]
    tangent_z[~idx] = -invLen * normals_y[~idx]

    return tangent


def get_both_orthogonal(normals):
    # norm_zero = torch.norm(normals,dim=-1) < 1e-8
    # normals[...,2] += norm_zero
    tangentT = get_orthogonal(normals)
    tangentS = torch.cross(tangentT, normals, dim=-1)
    return tangentT, tangentS


if __name__ == "__main__":
    device = "cuda"
    normals = torch.randn(5, 6, 3, device=device)
    # normals = torch.zeros(5,6,3,device=device)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    tangentT, tangentS = get_both_orthogonal(normals)
    print(tangentT, tangentS)

    print(torch.max(torch.abs(torch.sum(normals * tangentS, dim=-1))))
    print(torch.max(torch.abs(torch.sum(tangentT * normals, dim=-1))))
    print(torch.max(torch.abs(torch.sum(tangentT * tangentS, dim=-1))))
