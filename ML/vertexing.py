import torch

def chi2(origins, versors, weights, fitted_vertex):
    """Compute the chi2 of a set of tracks with respect to a fitted vertex
    Based on the chi2 definition in Pattern recognition, tracking and vertex reconstruction in particle detectors, 2021, R. Frühwirth
    Sec 8.1.1.1 formula 8.1
    chi2 = \sum_i [(o_i - v) x v_i]^2
    NOTE: it misses tracks' errors!!

    Parameters
    ----------
    origins : torch.Tensor (nTracks, 3)
        Origins of the tracks
    versors : torch.Tensor (nTracks, 3)
        Versors of the tracks
    weights : torch.Tensor (nTracks)
        Weights of the tracks, i.e. probability of the track to be from the fitted vertex
    fitted_vertex : torch.Tensor (3)
        Fitted vertex position

    Returns
    -------
    torch.Tensor
        Chi2 of the fit
    """

    # If no tracks' weights are provided, every track has the same weight
    if weights is None:
        weights = torch.ones(origins.shape[0])

    # Computation of the previous formula
    ov = origins - fitted_vertex
    chi2 = torch.linalg.cross(ov, versors)
    chi2_norm = torch.linalg.norm(chi2, axis=1) ** 2
    # Multiplication by the probability of the track to be from the fitted vertex
    chi2_weighted = weights * chi2_norm
    # Returning the sum of each track's chi2
    return torch.sum(chi2_weighted)

def fit(origins, versors, weights, fit_iter=100):
    """Fit a vertex to a set of tracks

    Parameters
    ----------
    origins : torch.Tensor (nTracks, 3)
        Origins of the tracks
    versors : torch.Tensor (nTracks, 3)
        Versors of the tracks
    weights : torch.Tensor (nTracks)
        Weights of the tracks, i.e. probability of the track to be from the fitted vertex
    fit_iter : int, optional
        Number of iterations of the fitting, by default 100

    Returns
    -------
    torch.Tensor, torch.Tensor
        Chi2 of the fit, fitted vertex
    """

    # fitted vertex tensor
    fitted_vertex = torch.tensor([0.,0.,0.], requires_grad=True)
    # Adam optimizer to minimize the chi2
    minimizer = torch.optim.Adam([fitted_vertex], lr=0.01)

    # Minimization loop
    for _ in range(fit_iter):
        minimizer.zero_grad()
        loss = chi2(origins, versors, weights, fitted_vertex)
        loss.backward(retain_graph=True)
        minimizer.step()

    return loss, fitted_vertex

def batch_vertex_loss(part_origin, part_versor, pred_particle, padding_mask,
                      fitted_vtx_ref, best_chi2_ref, fit_iter=100):
    """
    ...
    fitted_vtx_ref : torch.Tensor [B, 3]  — best-fit vertices from the dataloader
    best_chi2_ref  : torch.Tensor [B]     — best chi2 values from the dataloader
    """
    B = part_origin.shape[0]
    vtx_loss_total = torch.tensor(0.0, device=part_origin.device)
    n_valid_jets = 0

    for b in range(B):
        valid = ~padding_mask[b]

        if valid.sum() == 0:
            continue

        origins_b = part_origin[b][valid]
        versors_b = part_versor[b][valid]
        weights_b = torch.softmax(pred_particle[b][valid], dim=-1)[:, 1]

        # chi2 at the reference vertex, with current predicted weights
        current_chi2 = chi2(origins_b, versors_b, weights_b, fitted_vtx_ref[b])

        # Delta w.r.t. the best achievable chi2 — removes the zero-weight trivial minimum
        # clamp to avoid negative values from numerical noise
        vtx_loss_b = torch.clamp(current_chi2 - best_chi2_ref[b], min=0.0)

        vtx_loss_total = vtx_loss_total + vtx_loss_b
        n_valid_jets += 1

    if n_valid_jets == 0:
        return torch.tensor(0.0, device=part_origin.device)

    return vtx_loss_total / n_valid_jets