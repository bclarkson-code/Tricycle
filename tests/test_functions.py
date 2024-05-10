from tricycle.functions import sigmoid, tanh
from tricycle.tensor import to_tensor

# TODO: get these tests working
# These tests were built when there were multiple softmax candidates. They
# need to be updated when softmax gets optimised


# def test_softmax():
#     in_tensor = [np.pi, 0, -1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2
#     in_tensor_1 = to_tensor(in_tensor)
#     in_tensor_2 = to_tensor(in_tensor)
#
#     out_tensor = softmax(in_tensor_1)
#     # I am confident that v3 is correct, but slow so we'll use that as the
#     # source of truth
#     correct_out = softmax_v3(in_tensor_2)
#
#     assert out_tensor.close_to(correct_out, atol=1e-7)
#     out_tensor.backward()
#     correct_out.backward()
#
#     # If you work through the maths, the output gradient with an incoming
#     # gradient of all 1's is 0
#     assert in_tensor_1.grad.close_to(in_tensor_2.grad, atol=1e-6)
#
#
# def test_2d_softmax():
#     in_tensor = [
#         [np.pi, 0, -1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2
#     ] * 32
#     in_tensor_1 = to_tensor(in_tensor)
#     in_tensor_2 = to_tensor(in_tensor)
#
#     out_tensor = softmax_v2(in_tensor_1)
#     # I am confident that v3 is correct, but slow so we'll use that as the
#     # source of truth
#     correct_out = softmax_v3(in_tensor_2)
#
#     assert out_tensor.close_to(correct_out)
#     out_tensor.backward()
#     correct_out.backward()
#
#     assert in_tensor_1.grad.close_to(in_tensor_2.grad, atol=1e-6)
#
#
# def test_3d_softmax():
#     in_tensor = [
#         [[np.pi, 0, -1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 16
#     ] * 16
#     in_tensor_1 = to_tensor(in_tensor)
#     in_tensor_2 = to_tensor(in_tensor)
#
#     out_tensor = softmax_v2(in_tensor_1)
#     # I am confident that v3 is correct, but slow so we'll use that as the
#     # source of truth
#     correct_out = softmax_v3(in_tensor_2)
#
#     assert out_tensor.close_to(correct_out)
#     out_tensor.backward()
#     correct_out.backward()
#
#     assert in_tensor_1.grad.close_to(in_tensor_2.grad, atol=1e-6)
#
#
# def test_binary_softmax():
#     in_tensor = [[[1.0] + [0.0] * 15] * 16]
#     in_tensor_1 = to_tensor(in_tensor).to_gpu(1)
#     in_tensor_2 = to_tensor(in_tensor).to_gpu(1)
#
#     out_tensor = softmax_v2(in_tensor_1)
#     # I am confident that v3 is correct, but slow so we'll use that as the
#     # source of truth
#     correct_out = softmax_v3(in_tensor_2)
#
#     assert out_tensor.close_to(correct_out)
#     out_tensor.backward()
#     correct_out.backward()
#
#     assert in_tensor_1.grad.close_to(in_tensor_2.grad, atol=1e-6)


def test_sigmoid():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = sigmoid(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0.5, 0.73105858, 0.88079708, 0.95257413])

    out_tensor.backward()
    correct_grad = out_tensor * (1 - out_tensor)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct_grad)


def test_tanh():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = tanh(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0, 0.76159416, 0.96402758, 0.99505475])

    out_tensor.backward()
    correct_grad = 1 - out_tensor**2

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct_grad)
