import itertools
from utils import mnist_normalize
from utils import *

class JSMASolver:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def compute_jacobian(self, x: torch.Tensor):
        new_x = x.clone().requires_grad_(True)
        # Use softmax probabilities (not log-softmax) — JSMA saliency conditions
        # require ∂p_c/∂x_i, and log-softmax Jacobians don't satisfy them correctly
        probs = torch.exp(self.model(mnist_normalize(new_x)))
        num_classes = probs.shape[1]
        jacobian = torch.zeros(num_classes, new_x[0].numel(), device=new_x.device, dtype=new_x.dtype)
        for c in range(num_classes):
            grads = torch.autograd.grad(probs[0, c], new_x, retain_graph=True)[0]
            jacobian[c] = grads[0].reshape(-1)
        
        return jacobian

    def get_alpha_tensor(self, jacobian: torch.Tensor, target_class: int) -> torch.Tensor:
        return jacobian[target_class]

    def get_beta_tensor(self, jacobian: torch.Tensor, target_class: int) -> torch.Tensor:
        return jacobian.sum(dim=0) - jacobian[target_class]

    def do_saliency_extraction(self, jacobian: torch.Tensor, target_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.get_alpha_tensor(jacobian, target_class)
        beta = self.get_beta_tensor(jacobian, target_class)
        return alpha, beta    

    def get_inc_dec_scores(self, target: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inc_condition = (target > 0) & (other < 0)
        dec_condition = (target < 0) & (other > 0)
        increase = torch.where(inc_condition, target*torch.abs(other), torch.tensor(0.0))
        decrease = torch.where(dec_condition, torch.abs(target)*other, torch.tensor(0.0))
        return increase, decrease

    def compute_inc_dec_saliency_map(self, alpha: torch.Tensor, beta: torch.Tensor):
        inc_map, dec_map = self.get_inc_dec_scores(alpha, beta)
        return inc_map, dec_map

    
    def init_search_space(self, num_features: int):
        return torch.ones(num_features, dtype=bool)

    def prune_saturated_pixels(self, x: torch.Tensor, search_space: torch.Tensor, clip_min = 0.0, clip_max = 1.0):
        x_flat = x.flatten().clone()
        epsilon = 1e-6
        search_space = search_space & ~((x_flat <= clip_min + epsilon) | (x_flat >= clip_max - epsilon))
        return search_space

    def apply_perturbation(self, x: torch.Tensor, theta: float, p: int, q: int = None, clamp_min = 0.0, clamp_max = 1.0):
        # if increasing, theta = +theta; if decreasing, theta = -theta
        orig_shape = x.shape
        x_flat = x.flatten().clone()
        
        x_flat[p] = torch.clamp(
            x_flat[p] + theta,
            clamp_min, clamp_max
        )

        if q is not None:
            x_flat[q] = torch.clamp(
                x_flat[q] + theta,
                clamp_min, clamp_max
            )

        x_modified = x_flat.reshape(orig_shape)
        return x_modified


    def _get_inc_dec_pairwise_score(self, alpha: torch.Tensor, beta: torch.Tensor, p: int, q: int):
        alpha_pq = (alpha[p] + alpha[q]).item()
        beta_pq = (beta[p] + beta[q]).item()

        inc_score = alpha_pq * abs(beta_pq) if (alpha_pq > 0) and (beta_pq < 0) else 0.0
        dec_score = beta_pq * abs(alpha_pq) if (alpha_pq < 0) and (beta_pq > 0) else 0.0

        score = max(inc_score, dec_score)
        direction = None
        if score > 0:
            direction = 1 if inc_score > dec_score else -1
        
        return score, direction

        

    def compute_pairwise_saliency(self, alpha: torch.Tensor, beta: torch.Tensor, p: int, q: int, best_p: int, best_q: int, best_score: float, best_direction: int):

        score, direction = self._get_inc_dec_pairwise_score(alpha, beta, p, q)

        if direction is not None and score > best_score:
            best_score = score
            best_direction = direction
            best_p = p
            best_q = q
            
        return best_p, best_q, best_score, best_direction
        
    
    def get_top_k(self, alpha, beta, search_space, tk=256):
        valid = torch.nonzero(search_space).squeeze(1)
        valid_scores = torch.abs(alpha[valid]) * torch.abs(beta[valid])
        _, top_inds = torch.topk(valid_scores, k=min(tk, len(valid)))
        top_k = valid[top_inds]
        return top_k
        
        
    def _exceeds_l0_budget(self, x_orig: torch.Tensor, x_adv: torch.Tensor, budget: int):
        pixels_modified = (torch.abs(x_adv - x_orig) > 1e-6).sum().item()
        print(f"L0 budget exceeded: {pixels_modified >= budget}")
        return pixels_modified >= budget
    
    def _fools_model(self, model: torch.nn.Module, x_adv: torch.Tensor, target: int):
        with torch.no_grad():
            pred = model(mnist_normalize(x_adv)).argmax()
        print(f"Model fooled: {pred == target}")
        return pred == target

    def stop_search(self, model: torch.nn.Module, x_orig: torch.Tensor, x_adv: torch.Tensor, budget: int, target: int):
        return self._exceeds_l0_budget(x_orig, x_adv, budget) or self._fools_model(model, x_adv, target)
        

    def _do_pairwise_attack(self, search_space: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
            
        top_k = self.get_top_k(alpha, beta, search_space, tk=256)
        
        best_p, best_q, best_score, best_direction = -1, -1, -1.0, None
        for p, q in itertools.combinations(top_k.tolist(), 2):
            best_p, best_q, best_score, best_direction = self.compute_pairwise_saliency(alpha, beta, p, q, best_p, best_q, best_score, best_direction)
        
        if best_direction is None:
            return None

        
        return best_p, best_q, best_direction
        
    def _do_single_pixel_attack(self, alpha: torch.Tensor, beta: torch.Tensor, search_space: torch.Tensor):
        
        inc_map, dec_map = self.compute_inc_dec_saliency_map(alpha, beta)
        valid = torch.nonzero(search_space).squeeze(1)
        if valid.size() == 0:
            return None
        valid_inc, valid_dec = inc_map[valid], dec_map[valid]
        
        best_inc_idx = valid_inc.argmax().item()
        best_dec_idx = valid_dec.argmax().item()

        inc_score = inc_map[valid[best_inc_idx]]
        dec_score = dec_map[valid[best_dec_idx]]
        if inc_score + dec_score <= 1e-9:
            return None
        direction = 1
        p = valid[best_inc_idx].item()
        q = None # for clarity and consistency with _do_pairwise_attack
        if inc_score < dec_score:
            direction = -1
            p = valid[best_dec_idx].item()
        
        return p, q, direction


    def jsma_targeted(self, x: torch.Tensor, target: int, budget: int, theta: float):
        print("Doing the attack.")
        x_adv = x.clone()

        search_space = self.init_search_space(x_adv.numel())
        search_space = self.prune_saturated_pixels(x_adv, search_space)
        while(not self.stop_search(self.model, x, x_adv, budget, target)):
            
            jacobian = self.compute_jacobian(x_adv)
            alpha, beta = self.do_saliency_extraction(jacobian, target)
            
            res = self._do_pairwise_attack(search_space, alpha, beta)
            if res is None:
                print("No valid pair, falling back to single pixel")
                res = self._do_single_pixel_attack(alpha, beta, search_space)
                if res is None:
                    return None
            p, q, direction = res
            pixels_modified = (torch.abs(x_adv - x) > 1e-6).sum().item()
            print(f"p={p}, q={q}, direction={direction}, pixels_modified={pixels_modified}")
            x_adv = self.apply_perturbation(x_adv, theta * direction, p, q)
            search_space = self.prune_saturated_pixels(x_adv, search_space)
        return x_adv.detach().cpu().numpy()
        
        
    

def main() -> None:
    host = os.getenv("BASE_URL")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default=host, help="Server base URL"
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional local path to weights; downloads from the server when omitted",
    )
    args = parser.parse_args()

    # Reach the challenge API
    chall = fetch_challenge(args.host)
    weights_path = args.weights if args.weights else None
    model = load_model(args.host, weights_path)

    # Local sanity prediction on clean image
    x = torch.from_numpy(chall.x01)
    clean_pred = int(torch.argmax(model(mnist_normalize(x)), dim=1).item())
    print(f"Original prediction: {clean_pred}, True label: {chall.original_label}")
    print(f"Target class: {chall.target_class}, L0 budget: {chall.l0_budget}")
    
    # Perform JSMA attack - Execution will fail here upon calling the stub
    solver = JSMASolver(model)
    x_adv = solver.jsma_targeted(
        x,
        chall.target_class,
        chall.l0_budget,
        theta=1.0,
    )

    # Verify result locally
    adv_pred = int(
        torch.argmax(model(mnist_normalize(torch.from_numpy(x_adv))), dim=1).item()
    )
    l0_used = int(np.sum(np.abs(x_adv - chall.x01) > 1e-6))
    print(
        json.dumps(
            {
                "l0_used": l0_used,
                "adv_pred": adv_pred,
                "target": chall.target_class,
                "clean_pred": clean_pred,
            },
            indent=2,
        )
    )

    # Submit for the flag
    x2d = x_adv[0, 0].cpu().numpy() if isinstance(x_adv, torch.Tensor) else x_adv[0, 0]
    b64 = png_from_x01(x2d)
    r = requests.post(f"{args.host}/submit", json={"image_b64": b64}, timeout=10)
    try:
        r.raise_for_status()
    except Exception:
        print("Server response:", r.text)
        raise
    print("Flag:", r.json().get("flag"))

if __name__ == "__main__":
    main()