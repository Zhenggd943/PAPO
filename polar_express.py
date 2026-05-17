from itertools import repeat

coeffs_list = [
(8.28721201814563, -23.595886519098837, 17.300387312530933),
(4.107059111542203, -2.9478499167379106, 0.5448431082926601),
(3.9486908534822946, -2.908902115962949, 0.5518191394370137),
(3.3184196573706015, -2.488488024314874, 0.51004894012372),
(2.300652019954817, -1.6689039845747493, 0.4188073119525673),
(1.891301407787398, -1.2679958271945868, 0.37680408948524835),
(1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
(1.875, -1.25, 0.375)
]

coeffs_list = [(a, b, c) for (a, b, c) in coeffs_list[: -1]] + [coeffs_list[-1]]

def polar_express_(args, G):
    assert len(G.shape) == 2

    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    hs = coeffs_list[: args.iteration_step] + list(
        repeat(coeffs_list[-1], args.iteration_step - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X