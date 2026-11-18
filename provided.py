from ccgen import ccd_parts, get_residual


def write_residual(path="ccd_residual.py"):
    terms = get_residual(ccd_parts).split(" + ")
    lines = [terms[0]] + [("- " + x[1:]) if x.startswith("-") else ("+ " + x) for x in terms[1:]]
    with open(path, "w") as fh:
        fh.write("import numpy as np\n\n\ndef residual(f, g, t):\n")
        fh.write("    N = t.shape[0]\n    o = slice(0, N)\n    v = slice(N, None)\n    return (\n")
        for line in lines:
            fh.write(f"        {line}\n")
        fh.write("    )\n")


if __name__ == "__main__":
    write_residual()
