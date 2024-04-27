def get_flow_between_frames(x, spynet):
    """
    Get flow between frames t and t+1 from x.

    Args:
    x (tensor): The input tensor with dimensions (batch_size, num_frames, channels, height, width).
    spynet (function): The function that calculates the flow between two sets of frames.

    Returns:
    tuple: A tuple containing backward flows and forward flows.
    """
    b, n, c, h, w = x.size()
    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    # Calculate backward flow
    flows_backward = spynet(x_1, x_2)
    flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(4))]

    # Calculate forward flow
    flows_forward = spynet(x_2, x_1)
    flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(4))]

    return flows_backward, flows_forward