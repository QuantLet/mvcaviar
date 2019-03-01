import subprocess


tasks = [
    ["python", "test_tf.py", str(t), str(t - 20)] for t in range(500, 360, -20)
]
procs = [subprocess.Popen(task) for task in tasks]
for i, proc in enumerate(procs):
    out, err = proc.communicate()
    if err is not None:
        print("ERROR for TASK {}".format(" ".join(tasks[i])))
        print(err)

