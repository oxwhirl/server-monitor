# Server Monitor
A simple script that allows parallel monitoring of multi-server (gpu and cpu) utilization.

If you see any issues, please [open an issue](../../issues/new), or even better, open fix the issue and [submit a pull request](../../compare).

## Usage

``` sh
$ cd [/path/to/server-monitor]
$ CONDA_ENV_NAME="$(cat ./environment.yml | grep 'name: ' | awk '{ print $2}')" && \
  conda activate base && \
  conda env remove -y --name "${CONDA_ENV_NAME}"; \
  conda env create -v -f ./environment.yml -n "${CONDA_ENV_NAME}" && \
  conda activate "${CONDA_ENV_NAME}"
$ GPU_MACHINES=(gandalf gimli) && python -m monitor "${GPU_MACHINES[@]}"
```

Should output something like:
```
  hostname   CPU %                GPU compute %
  ─────────────────────────────────────────────────
  gandalf    [|           2.5%]   0 1 2 3 4 5 6 7
  gimli      [|           1.8%]   0 1 2 3 4 5 6 7
```
