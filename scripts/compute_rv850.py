#!/usr/bin/env python3
import argparse
import re
import shutil
import calendar
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime

def load_config(path):
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip().strip('"')
    return config

def compute_rv(u, v, lat, lon):
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))
    R    = 6371000.0
    dvdx = np.gradient(v, axis=-1) / (R * np.cos(np.deg2rad(lat))[:, None] * dlon[None, :])
    dudy = np.gradient(u * np.cos(np.deg2rad(lat))[:, None], axis=-2) / (R * dlat[:, None])
    return dvdx - dudy

def in_range(filename, start_dt, end_dt):
    date_str = next(
        (m.group() for p in [r'\d{8}', r'\d{6}', r'\d{4}']
         for m in [re.search(p, Path(filename).name)] if m), None
    )
    if date_str is None:
        return False
    for fmt in ['%Y%m%d', '%Y%m', '%Y']:
        try:
            return start_dt <= datetime.strptime(date_str, fmt) <= end_dt
        except ValueError:
            continue
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    config  = load_config(args.config)
    workdir = Path(args.workdir)
    outdir  = workdir / "preproc_RV850"
    outdir.mkdir(parents=True, exist_ok=True)

    for tmp in outdir.glob("tmp_*.nc"):
        tmp.unlink()

    start_dt = datetime.strptime(config["START"], "%Y%m")
    end_ym   = datetime.strptime(config["END"], "%Y%m")
    last_day = calendar.monthrange(end_ym.year, end_ym.month)[1]
    end_dt   = end_ym.replace(day=last_day)

    level_str = config["U850_LEVEL"]
    if level_str.strip():
        level_val = float("".join(filter(str.isdigit, level_str)))
        if "Pa" in level_str and "hPa" not in level_str:
            level_val /= 100.0
    else:
        level_val = None

    u_files = sorted(
        f for f in Path(config["U850_DIR"]).glob("*.nc*")
        if in_range(f.name, start_dt, end_dt)
    )

    for ufile in u_files:
        date_str = next(
            (m.group() for p in [r'\d{8}', r'\d{6}', r'\d{4}']
             for m in [re.search(p, ufile.name)] if m), None
        )
        if date_str is None:
            continue

        v_matches = sorted(Path(config["V850_DIR"]).glob(f"*{date_str}*"))
        if not v_matches:
            print(f"WARNING: No V file for {date_str}, skipping")
            continue

        outfile = outdir / f"rv850_{date_str}.nc"
        if outfile.exists():
            print(f"  exists: {outfile}")
            continue

        print(f"  processing: {date_str}")

        ds_u = xr.open_dataset(ufile)
        ds_v = xr.open_dataset(v_matches[0])

        if level_val is not None:
            u = ds_u[config["U850"]].sel(lev=level_val, method="nearest")
            v = ds_v[config["V850"]].sel(lev=level_val, method="nearest")
        else:
            u = ds_u[config["U850"]]
            v = ds_v[config["V850"]]

        lat = u[config["LATNAME"]].values
        lon = u[config["LONNAME"]].values

        rv_data = xr.apply_ufunc(
            compute_rv,
            u, v,
            kwargs={"lat": lat, "lon": lon},
            input_core_dims=[["lat", "lon"], ["lat", "lon"]],
            output_core_dims=[["lat", "lon"]],
            vectorize=True,
        )
        rv_vals = rv_data.values
        ds_u.close()
        ds_v.close()

        with Dataset(str(ufile)) as src, Dataset(str(outfile), "w") as dst:
            dst.createDimension("time", None)
            dst.createDimension("lat",  len(lat))
            dst.createDimension("lon",  len(lon))

            t_src    = src["time"]
            t_dst    = dst.createVariable("time", t_src.dtype, ("time",))
            t_dst.setncatts({k: t_src.getncattr(k) for k in t_src.ncattrs()})
            t_dst[:] = t_src[:]

            lat_dst    = dst.createVariable("lat", "f4", ("lat",))
            lat_dst[:] = lat
            lon_dst    = dst.createVariable("lon", "f4", ("lon",))
            lon_dst[:] = lon

            rv_dst           = dst.createVariable("vor", "f4", ("time", "lat", "lon"), zlib=True)
            rv_dst.units     = "s-1"
            rv_dst.long_name = "850hPa relative vorticity"
            rv_dst[:]        = rv_vals

        print(f"  computed: {outfile}")

    print(f"[DONE] RV850 written to {outdir}")

    updated_config = Path(args.config).parent / f"{config['DATANAME']}_updated.conf"
    if not updated_config.exists():
        shutil.copy(args.config, updated_config)
    with open(updated_config, "a") as f:
        f.write(f'\nRV850_DIR="{outdir.resolve()}"\n')
        f.write(f'RV850="vor"\n')
        f.write(f'RV850_LEVEL=""\n')
    print(f"[DONE] Updated config: {updated_config}")

if __name__ == "__main__":
    main()
