# Server Monitor
A simple script that allows parallel monitoring of multi-server gpu and cpu utilization.

The usage information is computed as follows:
* `CPU %` usage is computed as `[CPU load average] / [number of CPUs]` and presented as a colored bar like in htop (e.g. `[||         16.2%]`).
* `RAM %` usage is computed as `[RAM usage] / [total RAM]` and presented as a colored bar as described above for `CPU %`. Both `[RAM usage]` and `[total RAM]` are calculated based on the actual physical RAM, that is, they exclude swap memory.
* `GPU compute %` info column presents an array where each element correspond to a machine's GPU and the element's color indicates that GPU's compute utilization.

For all above, the colors change from `white -> green -> yellow -> orange -> red` corresponding to utilization thresholds of `<=0%, <=25%, <=50%, <=75%, <=100%`.

## Usage

``` sh
$ cd [/path/to/server-monitor]
$ CONDA_ENV_NAME="$(cat ./environment.yml | grep 'name: ' | awk '{ print $2}')" && \
  conda activate base && \
  conda env remove -y --name "${CONDA_ENV_NAME}"; \
  conda env create -v -f ./environment.yml -n "${CONDA_ENV_NAME}" && \
  conda activate "${CONDA_ENV_NAME}"
$ python setup.py develop
$ GPU_MACHINES=(gandalf gimli) && server-monitor "${GPU_MACHINES[@]}"
```

Should output something like:
```
  hostname   CPU %                RAM %                GPU compute %
 ──────────────────────────────────────────────────────────────────────
  gandalf    [|           2.5%]   [||         12.6%]   0 1 2 3 4 5 6 7
  gimli      [|           1.8%]   [||||       28.3%]   0 1 2 3 4 5 6 7
```

## Issues
If you see any issues, please [open an issue](../../issues/new), or even better, fix the issue and [submit a pull request](../../compare).
