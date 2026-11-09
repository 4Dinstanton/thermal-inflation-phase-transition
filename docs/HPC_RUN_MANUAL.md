# Thermal-inflation 격자 시뮬레이션 — 슈퍼컴 실행 매뉴얼

KISTI Nurion에서 가장 많이 돌린 **표준 조합(set8 production)**을 다른 센터에서
그대로 재현하기 위한 최소 안내입니다. 상세 CLI 전체는
[`cosmolattice_ext/RUNNING_AND_POTENTIAL.md`](../cosmolattice_ext/RUNNING_AND_POTENTIAL.md) 참고.

---

## 0. 표준 조합 (이것만 돌리면 됨)

| 항목 | 값 |
|------|----|
| 격자 | `Nx=1024` (1024³), `dx_phys=1e-3`, `dt_phys=1e-4` GeV⁻¹, `tMax=1000` (=10,000 step) |
| 퍼텐셜 | `V_correct`, `nb=nf=20`, `T0=1230` GeV, `γ=4.1667e-4` (`param_set=set8`) |
| 장 | complex (`--n_scalars 2`), on-lattice GW (`--with_gws`) |
| Langevin | `stochasticrk` (numba scheme), `η∝T` (`--eta_follows_T`), false-vac 비율 ≤ 0.01에서 global off |
| 팽창 | `staged` TI→MD→RD, `f_switch=0.01`, `φ_esc=5e4` GeV, `T_rh=1000` GeV |
| 출력 | 평균/스펙트럼/GW txt 매 10 step, 3D energy HDF5 매 200 step, 체크포인트 매 1000 step, 3D 장 스냅샷 4000 step(φ>5e4 GeV 이후 100 step) |

이 조합은 `simulation/run_tipt_gamma.sh`의 기본값입니다. 환경변수 `NX`, `NP`만 바꿔서 씁니다.

---

## 1. 준비물

- C++14 컴파일러 + MPI (KISTI: `gcc/8.3.0` + `openmpi/3.1.0`)
- **parallel** HDF5 (KISTI: `hdf5-parallel/1.10.2`), FFTW3 MPI 버전 (KISTI: `fftw_mpi/3.3.7`)
- `cmake` ≥ 3.12, `python3` ≥ 3.9 (실행 스크립트는 표준 라이브러리만 사용; 후처리는 `requirements.txt`)

```bash
git clone --recursive https://github.com/4Dinstanton/thermal-inflation-phase-transition.git PhaseTransition
cd PhaseTransition
git submodule update --init      # external/cosmolattice @ a5654d64 (고정 커밋; 패치 anchor가 이 버전 기준)
```

**`data/thermal_splines/thermal_tables.bin`은 git에 없습니다.** (`.gitignore`)
저희가 보내드리는 파일(196 KB)을 위 경로에 두거나, 직접 생성하세요
(`pip install numpy scipy cosmoTransitions` 후 `python3 tools/export_thermal_splines.py`).

---

## 2. 빌드 (계산 노드 아키텍처에서 1회)

로그인 노드와 계산 노드 CPU가 다르면 **계산 노드에서** 빌드해야 합니다
(Nurion에서 SKL 바이너리는 KNL에서 안 돌았음).

```bash
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <cmake> <python3>   # 센터 이름에 맞게
export CC=mpicc CXX=mpicxx CXXFLAGS="-std=c++14"
export HDF5_ROOT=/path/to/hdf5-parallel        # include/hdf5.h 가 있는 prefix
export MYFFTW3_PATH=/path/to/fftw-mpi          # include/fftw3-mpi.h 가 있는 prefix

rm -rf external/cosmolattice/build_mpi
python3 simulation/run_cosmolattice.py --install --build --mpi --dry_run \
  --Nx 32 --tMax 1 --no_snapshots
ls -l external/cosmolattice/build_mpi/thermal_inflation    # 이게 생기면 성공
```

`--install`이 CosmoLattice 서브모듈에 모델 헤더 symlink와 필요한 패치
(Langevin evolver 등록, 스냅샷 hook, HDF5 문자열 길이 256→1024 등)를 자동으로 넣습니다.
서브모듈을 손으로 고칠 필요는 없습니다. KISTI용 빌드 job 예시는 `simulation/pbs_build_tipt_skl.pbs`에 있습니다.

---

## 3. 스모크 테스트 (몇 분)

256³ 이하에서는 raw 스냅샷을 NPZ로 변환하므로 `numpy`가 필요합니다.

```bash
NX=64 NP=8 TMAX=20 bash simulation/run_tipt_gamma.sh
tail data/lattice/set8/64x64x64_*_CL/run.log    # "Step N done. ... T=... GeV" 줄이 찍히면 정상
```

---

## 4. Production 실행

### 표준 CLI (스크립트 내부에서 실제로 실행되는 것)

```bash
python3 simulation/run_cosmolattice.py \
  --mpi --np "$NP" --Nx 1024 --tMax 1000 \
  --dx_phys 1e-3 --dt_phys 1e-4 --T0 1230 --gamma 4.1667e-4 --param_set set8 \
  --potential_type V_correct --nb 20 --nf 20 \
  --eta_follows_T --n_scalars 2 --with_gws \
  --expansion_mode staged --expansion_f_switch 0.01 --expansion_phi_esc 5e4 --T_rh 1000 \
  --langevin_off_after_nucleation --langevin_off_f_switch 0.01 --langevin_off_phi_esc 5e4 \
  --phi_threshold 50000 --steps 4000 --steps_dense 100 \
  --tOutputFreq 1 --tOutputInfreq 1 --tOutputRareFreq 20 --backup_steps 1000 \
  --snapshot_format hdf5
```

같은 실행을 한 줄로 하면 `NX=1024 NP=<rank 수> bash simulation/run_tipt_gamma.sh`입니다.
`mpirun`은 `PATH`의 것을 사용하고, 반드시 **repo 루트에서** 실행해야 합니다
(`input.in`의 출력 경로가 repo 기준 상대경로).

### Slurm 템플릿

```bash
#!/bin/bash
#SBATCH -J tipt1024
#SBATCH -N 16
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -t 96:00:00
#SBATCH -p <partition>
#SBATCH -o tipt1024.%j.log

REPO=/path/to/PhaseTransition
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <python3>
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
NX=1024 NP=$SLURM_NTASKS bash simulation/run_tipt_gamma.sh
```

### PBS 템플릿 (KISTI에서 쓰던 형태)

```bash
#!/bin/bash
#PBS -N tipt1024
#PBS -q <queue>
#PBS -l select=16:ncpus=64:mpiprocs=64:ompthreads=1
#PBS -l walltime=96:00:00
#PBS -j oe

REPO=/path/to/PhaseTransition
module load <gcc> <openmpi> <hdf5-parallel> <fftw-mpi> <python3>
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
NX=1024 NP=1024 bash simulation/run_tipt_gamma.sh
```

`mpirun` 대신 `srun`만 허용되는 센터라면 job 안에서 `DRY_RUN=1 NX=1024 NP=... bash simulation/run_tipt_gamma.sh`로
`input.in`만 만든 뒤 직접 실행하세요.

```bash
srun external/cosmolattice/build_mpi/thermal_inflation input=data/lattice/set8/<run_dir>/input.in
```

---

## 5. 자원 산정 (KISTI 실측)

| 구성 | 결과 |
|------|------|
| 1024³, 64 rank / KNL 1노드 | 메모리 초과(OOM)로 kill |
| 1024³, 128 rank / SKL 4노드 (노드당 192 GB, 총 768 GB) | cgroup 메모리 초과로 kill |
| 1024³, **1024 rank** (KNL, 노드당 64 rank) | 정상 동작. 약 23–27 s/step, **48 h 동안 t=642(6,420 step)까지만 진행** |

- **메모리:** 총 768 GB로는 부족했습니다. 1.5 TB 이상을 잡으세요.
- **도메인 분할:** slab 방식(`(NP 1 1)`)입니다. `NP`는 1024의 약수(256, 512, 1024)로 하고 1024를 넘기지 마세요.
- **Walltime:** KNL 기준으로 10,000 step 전체에 약 75 h가 걸립니다. 최신 Xeon/EPYC는 더 빠르겠지만 **≥ 96 h 또는 long 큐**를 권장합니다.
- **재시작:** CosmoLattice 체크포인트(`thermal_inflation_backup.h5`, 1000 step마다)는 쓰이지만,
  모델의 staged 팽창 단계·온도·Langevin-off 상태는 백업에 저장되지 않습니다.
  재시작은 **검증되지 않았으므로** 한 번에 끝까지 돌리는 것을 전제로 하세요.
- **디스크:** 1024³ 실수장 1개가 8 GiB입니다. 3D 스냅샷(장 2성분 + energy 3종)이 반복적으로 쌓이므로 scratch를 **수 TB** 확보하세요.

---

## 6. 출력 & 모니터링

출력 디렉터리는 `data/lattice/set8/1024x1024x1024_T0_1230_complex_..._etaT_langoff_f0.01_..._stochasticrk_V_correct_CL/`입니다.

| 파일 | 내용 |
|------|------|
| `run.log` | 실시간 진행 (`Step N done. Current time: t  T=... GeV`) |
| `average_energies.txt`, `average_scalar_*.txt`, `average_scale_factor.txt` | 평균량 (매 10 step) |
| `spectra_scalar_*.txt`, `spectra_gws.txt`, `energy_gws.txt` | 스펙트럼 / GW (매 10 step) |
| `field_snapshot.h5`, `*_snapshot*.h5` | 3D 장 / 에너지 (대용량) |
| `input.in`, `cl_run_params.json` | 실제 사용된 파라미터 |

```bash
tail -f data/lattice/set8/<run_dir>/run.log
export TIPT_LIVE_LOG=/scratch/<user>/tipt_live.log   # 선택: job 시작 전에 설정하면 이 파일에도 실시간 기록
```

저희에게 결과를 보낼 때는 **txt 파일 전부와 `input.in`, `run.log`**를 우선 보내주세요.
`.h5`는 따로 협의합니다. GW 스펙트럼은
`python3 postprocess/plot_cl_gw_spectrum.py <run_dir> --physical --T_rh 1000`로 그립니다.
1024³ string 분석(`tools/cl_hdf5_string_pipeline.py`)에는 노드당 128 GB 이상 RAM이 필요합니다.

---

## 7. 변형 (필요할 때만)

| 목적 | 방법 |
|------|------|
| fermion-only 퍼텐셜 | `TIPT_POT=fermion_only` (T0=1600 자동) |
| 다른 γ | `GAMMA=1e-5` → `data/lattice/set_gamma_1e-05/` (set8을 덮어쓰지 않음) |
| 256³ 빠른 버전 | `NX=256 NP=64` (스냅샷은 자동으로 raw) |
| 명령만 확인 | `DRY_RUN=1` |

**주의:** `DRY_RUN=1`이나 같은 조합으로 재실행하면 같은 이름의 기존 run 디렉터리에 있는 `input.in`과 `cl_run_params.json`을 덮어씁니다.
기존 결과를 보존하려면 4절의 CLI를 직접 쓰면서 `--run_name <이름>`을 붙이거나, run 디렉터리를 먼저 옮겨 두세요.
