import pytest
import torch

from vision_repro.relu_redirect import (
    LucidRedirectReLU,
    RedirectState,
    install_redirected_relu,
)


def backward(values, grads, step=0):
    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    state = RedirectState(global_step=step, record=True)
    out = LucidRedirectReLU(state)(x)
    out.backward(torch.tensor(grads, dtype=torch.float32))
    return out.detach(), x.grad, state


def test_blocked_batch_and_descent_sign():
    out, gradient, state = backward([[-2, -3], [-2, -3]], [[-1, 2], [1, -4]])
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(gradient, torch.tensor([[-1, 0], [0, -4]]))
    assert state.opened_batch_ops == 2
    assert state.passed_negative_elements == 2


def test_any_normal_gradient_prevents_opening_whole_batch_entry():
    _, gradient, _ = backward([[-2, 3], [-2, 3]], [[-1, -1], [-1, 0]])
    assert torch.equal(gradient, torch.tensor([[0, -1], [-1, 0]]))


@pytest.mark.parametrize("step,expected", [(16, -1), (17, 0)])
def test_exact_source_global_step_boundary(step, expected):
    _, gradient, _ = backward([[-2]], [[-1]], step)
    assert gradient.item() == expected


@pytest.mark.parametrize("step,expected", [(15, -1), (16, 0)])
def test_explicit_sixteen_updates_boundary(step, expected):
    state = RedirectState(global_step=step, last_global_step=15)
    x = torch.tensor([[-2.0]], requires_grad=True)
    (-LucidRedirectReLU(state)(x).sum()).backward()
    assert x.grad.item() == expected


def test_zero_boundary_matches_source():
    _, gradient, _ = backward([[0, -2]], [[-1, -1]])
    assert torch.equal(gradient, torch.tensor([[-1, 0]]))


def test_forward_and_model_isolation_and_intermediate_relu():
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 1, bias=False))
    other = torch.nn.Sequential(torch.nn.ReLU())
    with torch.no_grad():
        model[1].weight.fill_(1)
    state = RedirectState()
    before = model(torch.tensor([[-1.0, -2.0]])).detach()
    assert install_redirected_relu(model, state) == ["0"]
    assert isinstance(other[0], torch.nn.ReLU)
    x = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    assert torch.equal(before, model(x).detach())
    (-model(x).sum()).backward()
    assert torch.equal(x.grad, torch.tensor([[-1.0, -1.0]]))


def test_explicit_functional_relu_wrapper_type():
    class ReluWrapper(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.relu(value)

    model = torch.nn.Sequential(ReluWrapper())
    assert install_redirected_relu(model, RedirectState(), (ReluWrapper,)) == ["0"]
    x = torch.tensor([[-1.0]], requires_grad=True)
    (-model(x).sum()).backward()
    assert x.grad.item() == -1


@pytest.mark.parametrize("step", [0, 16, 17])
def test_matches_literal_source_rule_on_bchw(step):
    torch.manual_seed(12)
    x = torch.randn(4, 3, 5, 5, requires_grad=True)
    upstream = torch.randn_like(x)
    with torch.no_grad():
        x[0].fill_(-1)
        upstream[1].zero_()
    normal = torch.where(x < 0, torch.zeros_like(upstream), upstream)
    redirected = torch.where((x < 0) & (upstream > 0), 0.0, upstream)
    mask = (normal.flatten(1).norm(dim=1) > 0).reshape(-1, 1, 1, 1)
    expected = normal if step > 16 else torch.where(mask, normal, redirected)
    actual = LucidRedirectReLU(RedirectState(global_step=step))(x)
    assert torch.equal(actual, torch.relu(x))
    actual.backward(upstream)
    assert torch.equal(x.grad, expected)
