from marslab.imgops.look import Look


def test_look():
    """
    Very basic test for Look.
    """
    from matplotlib.figure import Figure
    import numpy as np

    from marslab.imgops import render
    from marslab.imgops import masking
    from marslab.imgops import imgutils

    def enhancer(arrs, mask):
        masked = np.ma.masked_array(
            np.dstack(arrs), mask=np.tile(mask, (1, 1, 3))
        )
        return imgutils.enhance_color(masked, (0, 1), 1)

    list_ = []
    instruction = {
        "crop": (1, 1, 1, 1),
        "prefilter": {
            "function": imgutils.normalize_range,
            "params": {"stretch": 2},
        },
        "look": enhancer,
        "mask": {
            "instructions": [
                {
                    "function": masking.threshold_mask,
                    "params": {"percentiles": (55, 95)},
                    "colorfill": {"color": 0, "mask_alpha": 1},
                    "pass": True,
                    "send": True,
                }
            ]
        },
        "plotter": {"function": render.simple_figure},
        "bang": {"function": lambda: list_.append(1)}
    }
    look = Look.compile_from_instruction(instruction)
    r = np.random.random((128, 128))
    g = np.random.random((128, 128)) * 10
    b = np.random.random((128, 128)) * 100
    fig = look.execute((r, g, b))
    assert isinstance(fig, Figure)
    assert look.inserts['plotter']['layers'][0][0].shape == (126, 126, 4)
    assert list_ == [1]
