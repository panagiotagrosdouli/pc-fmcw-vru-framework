import subprocess, sys

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    run([sys.executable, "run_part_a.py"])
    run([sys.executable, "run_part_b.py"])
    print("\nDone. See figures/ and outputs/.")
