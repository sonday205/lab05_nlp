import torch
from torch import nn


class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        bound = hidden_size ** -0.5
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)

    def forward(self, x_t, h_prev):
        return torch.tanh(x_t @ self.weight_ih.T + h_prev @ self.weight_hh.T + self.bias)


def check_framework_parity():
    torch.manual_seed(123)
    custom = SimpleRNNCell(3, 4).double()
    reference = nn.RNNCell(3, 4, nonlinearity="tanh").double()
    with torch.no_grad():
        reference.weight_ih.copy_(custom.weight_ih)
        reference.weight_hh.copy_(custom.weight_hh)
        reference.bias_ih.copy_(custom.bias)
        reference.bias_hh.zero_()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    actual, expected = custom(x, h), reference(x, h)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, h))
    expected_grad = torch.autograd.grad(expected.square().sum(), (x, h))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    return {"output_max_abs_error": (actual - expected).abs().max().item(),
            "input_gradient_max_abs_error": (actual_grad[0] - expected_grad[0]).abs().max().item(),
            "hidden_gradient_max_abs_error": (actual_grad[1] - expected_grad[1]).abs().max().item(),
            "batch_size": 2, "input_size": 3, "hidden_size": 4}


if __name__ == "__main__":
    print(check_framework_parity())
