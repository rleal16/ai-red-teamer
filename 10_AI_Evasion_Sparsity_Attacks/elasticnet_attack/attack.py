from utils import *
from utils import _post_submit_retry, _post_submit

class EAD:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        beta: float,
        elastic_max: float,
        l2_max: float,
        l1_max: float,
        max_iter: int = 400,
        lr: float = 1e-2,
        bin_steps: int = 5,
        initial_const: float = 1e-3,
    ) -> None:
        self.model = model
        self.device = device
        self.beta = beta
        self.max_iter = max_iter
        self.lr = lr
        self.bin_steps = bin_steps
        self.initial_const = initial_const
        self.confidence = 0.0
        self.orig = None
        self.elastic_max = elastic_max
        self.l2_max = l2_max
        self.l1_max = l1_max


    def _check_constraints(self, l1: torch.Tensor, l2: torch.Tensor, elastic: torch.Tensor):
        return (self._check_l1_constraint(l1) & self._check_l2_constraint(l2) & self._check_elastic_constraint(elastic)) 
    def _check_elastic_constraint(self, elastic: torch.Tensor):
        return elastic <= self.elastic_max
    def _check_l2_constraint(self, l2: torch.Tensor):
        return l2 <= self.l2_max
    def _check_l1_constraint(self, l1: torch.Tensor):
        return l1 <= self.l1_max

    def _loss(self, x: torch.Tensor, y_onehot: torch.Tensor, const: torch.Tensor):
        
        logits = self.model(mnist_normalize(x))
        inv_onehot = (1-y_onehot)*logits-y_onehot*1e4
        real = torch.sum(y_onehot*logits, dim=1)
        other = torch.max(inv_onehot, dim=1)[0]
        adv_loss = torch.clamp(real - other + self.confidence, min=0)
        delta = x - self.orig
        delta_flat = delta.view(delta.size(0), -1)
        l1 = delta_flat.abs().sum(1)
        l2 = (delta_flat ** 2).sum(dim=1)
        elastic_dist = l2 + self.beta*l1
        total = torch.sum(const * adv_loss) + torch.sum(l2) + torch.sum(self.beta*l1)

        return total, l1, l2, elastic_dist


    def _prox(self, x: torch.Tensor, y: torch.Tensor, step: int):
        delta = y - self.orig
        upper = delta > self.beta
        within = delta.abs() <= self.beta
        lower = delta < -self.beta
        upper_values = torch.clamp(y - self.beta, 0, 1)
        lower_values = torch.clamp(y + self.beta, 0, 1)
        x_new = upper_values * upper.float() + within.float() * self.orig + lower_values * lower.float()
        zt = step/(step + 3.0)
        y_new = x_new + zt * (x_new - x)

        return x_new.clamp(0, 1), y_new.clamp(0, 1)

    def _update_const(self, const: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, model_fooled : torch.Tensor):
        """
        returns new lower, upper, and const tensors
        """
        
        new_lower = torch.where(model_fooled, lower, const)
        new_upper = torch.where(model_fooled, const, upper)
        midpoint = (new_upper + new_lower) / 2
        huge = upper > 1e9
        value_if_not_fooled = torch.where(huge, const*10, midpoint)
        
        new_const = torch.where(
            model_fooled, 
            midpoint,
            value_if_not_fooled
            )
        
        return new_lower, new_upper, new_const
        
    def _model_fooled(self, model: torch.nn.Module, x_adv : torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            logits = model(mnist_normalize(x_adv))
            pred = logits.argmax(dim=1)

        return pred != y.view_as(pred)




    def _labels_to_onehot(self, y: torch.Tensor, bsz: int):
        y_onehot = torch.zeros(bsz, 10, device=self.device)
        v = y.view(-1, 1)
        y_onehot.scatter_(1, v, 1)
        return y_onehot
    
    def _get_init_bin_search_values(self, bsz: int):
        const = torch.ones(bsz, device=self.device) * self.initial_const
        lower = torch.zeros(bsz, device=self.device)
        upper = torch.ones(bsz, device=self.device)*1e10
        return const, lower, upper

    def _update_bests(self, model_fooled: torch.Tensor, curr_adv: torch.Tensor, elastic_dist: torch.Tensor, best_adv: torch.Tensor, best_elastic: torch.Tensor):
         
        update_bests = (model_fooled) & (best_elastic >= elastic_dist)
        update_bests_4d = update_bests.view(-1, 1, 1, 1)
        best_adv = torch.where(update_bests_4d, curr_adv, best_adv)
        best_elastic = torch.where(update_bests, elastic_dist, best_elastic)

        return best_adv, best_elastic

    def run(self, x, y):
        self.orig = x.clone().to(self.device)
        
        batch_size = x.shape[0]
        print(f"Batch size type: {type(batch_size)}")
        y_onehot = self._labels_to_onehot(y, batch_size)
        const, lower, upper = self._get_init_bin_search_values(batch_size)
        best_adv = self.orig.clone().detach()
        best_elastic = torch.ones(batch_size, device = self.device) * torch.inf
        was_inner_fooled = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        for i in range(self.bin_steps):
            was_inner_fooled = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            x = self.orig.clone().detach()
            z = self.orig.clone().detach()
            for it in range(self.max_iter):
                
                z.requires_grad_(True)
                total, l1, l2, elastic_dist = self._loss(z, y_onehot, const)
                grads = torch.autograd.grad(total, z)[0]
                with torch.no_grad():
                    z -= self.lr*grads
                    x, z = self._prox(x, z, it)
                    
                if it % 100 == 0:
                    inner_fooled = self._model_fooled(self.model, x, y)
                    was_inner_fooled |= inner_fooled
                    best_adv, best_elastic = self._update_bests(inner_fooled, x, elastic_dist, best_adv, best_elastic)
            outer_fooled = self._model_fooled(self.model, x, y)
            
            model_fooled = outer_fooled | was_inner_fooled
            best_adv, best_elastic = self._update_bests(model_fooled, x, elastic_dist, best_adv, best_elastic)
            lower, upper, const = self._update_const(const, lower, upper, outer_fooled)
         
        
        return best_adv



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=BASE_URL)
    parser.add_argument("--weights", default="elasticnet_weights.pth")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--bin-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chall = fetch_challenge(args.host)
    model = load_model(args.weights, device)
    elastic_max = chall.elastic_max
    l2_max = chall.l2_max
    l1_max = chall.l1_max
    print(f"beta: {chall.beta}, elastic_max: {chall.elastic_max}, l2_max: {chall.l2_max}, l1_max: {chall.l1_max}")
    x = torch.from_numpy(chall.x01).to(device)
    y = torch.tensor([chall.label], device=device, dtype=torch.long)

    clean_pred = int(torch.argmax(model(mnist_normalize(x)), dim=1).item())
    ead = EAD(
        model,
        device,
        beta=chall.beta,
        max_iter=args.max_iter,
        lr=args.lr,
        bin_steps=args.bin_steps,
        elastic_max=elastic_max,
        l2_max=l2_max,
        l1_max=l1_max
    )
   
    adv = ead.run(x, y)
    
    with torch.no_grad():
        adv_pred = int(torch.argmax(model(mnist_normalize(adv)), dim=1).item())
    diff = adv.detach().cpu().numpy() - x.detach().cpu().numpy()
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff**2)))
    linf = float(np.max(np.abs(diff)))
    elastic = l2 + chall.beta * l1
    print(
        json.dumps(
            {
                "clean_pred": clean_pred,
                "adv_pred": adv_pred,
                "l1": l1,
                "l2": l2,
                "linf": linf,
                "elastic": elastic,
                "beta": chall.beta,
            },
            indent=2,
        )
    )

    b64 = b64_png_from_x01(adv.detach().cpu().numpy()[0, 0])
    r = _post_submit_retry(args.host, b64)
    try:
        r.raise_for_status()
    except Exception:
        print("Server response:", r.text)
        raise
    print("Flag:", r.json().get("flag"))


if __name__ == "__main__":
    main()