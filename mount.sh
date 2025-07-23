#!/bin/bash
#
# Sets up mounts for torchx jobs:
# - twmount to set up NFS
# - oilfs for warm storage mounts from the appropriate region
# - oilfs for airstore if enabled
# - manifoldfs for the provided Manifold bucket
# - manifuse (oilfs-mf) for the provided Manifold bucket
#
# This script depends on oilfs, airstore, twmount and mainfoldfs being
# available with the job.
#
# Pass in DISABLE_NFS=1, DISABLE_OILFS=1, DISABLE_MANIFOLDFS=1, or DISABLE_MANIFUSE=1
# to disable the respective filesystems. Otherwise jobs will fail explicitly.

export PS4=' + [$(date +"%Y-%m-%d %H:%M:%S,%3N")] '
set -eExu -o pipefail

# want REGION_DATACENTER_PREFIX
source /etc/fbwhoami

function mount_oilfs {
  local fuse_src_path="$1"
  local fuse_dst="$2"
  local fuse_src="${3-}"

  if [[ -n "${ENABLE_OILFS_NUMA_BINDING-}" ]]; then
    # Mount OILFS twice
    mount_oilfs_with_numa_binding "$fuse_src_path" "${fuse_dst}" "$fuse_src" "0"
    mount_oilfs_with_numa_binding "$fuse_src_path" "${fuse_dst}1" "$fuse_src" "1"
  else
    mount_oilfs_with_numa_binding "$fuse_src_path" "$fuse_dst" "$fuse_src"
  fi
}


#######################################
# Mount oilfs, introspecting on the host and setting the right source if required.
# Arguments
#   fuse_src_path path in ws://ws.ai.<region>0genai/<path> to mount
#   fuse_dst destination to mount to
#   fuse_src override path explicitly, ignoring the region entirely if set (optional)
#######################################
function mount_oilfs_with_numa_binding {
  local fuse_src_path="$1"
  local fuse_dst="$2"
  local fuse_src="${3-}"
  local numa_binding="${4-}"

  mkdir -p "${fuse_dst}"

  if [[ -z "$fuse_src" ]]; then
    local host
    host="$(hostname)"

    case $host in
      *.pci* )
        fuse_src="ws://ws.ai.pci0ai/${fuse_src_path}"
        ;;
      *.eag* )
        fuse_src="ws://ws.ai.eag0genai/${fuse_src_path}"
        ;;
      *.gtn* )
        fuse_src="ws://ws.ai.gtn0genai/${fuse_src_path}"
        ;;
      *.nha* )
        fuse_src="ws://ws.ai.nha0genai/${fuse_src_path}"
        ;;
      *.snb* )
        fuse_src="ws://ws.ai.snb0genai/${fuse_src_path}"
        ;;
      *.nao* )
        fuse_src="ws://ws.ai.nao0ai/${fuse_src_path}"
        ;;
      * )
        echo "No source available based on host $host. Can't mount OilFS!." 1>&2
        exit 1
        ;;
    esac
  elif [[ $fuse_src == ws://ws.ai.pci0genai/* ]]; then
    echo "Replacing deprecated ws://ws.ai.pci0genai cluster with ws://ws.ai.pci0ai" 1>&2
    echo "Original fuse_src: $fuse_src" 1>&2
    fuse_src=$(echo "$fuse_src" | sed 's/ws:\/\/ws.ai.pci0genai/ws:\/\/ws.ai.pci0ai/')
    echo "New fuse_src: $fuse_src" 1>&2
  fi

  [[ "${numa_binding}" ]] &&  export NUMA_BINDING="${numa_binding}"
  [[ "${numa_binding}" == "0" ]] && [[ "${OILFS_NUMA_NUM_CORES-}" == "12" ]] && export NUMA_PHYSBIND="0-5,112-117"
  [[ "${numa_binding}" == "1" ]] && [[ "${OILFS_NUMA_NUM_CORES-}" == "12" ]] && export NUMA_PHYSBIND="56-61,168-173"
  [[ "${numa_binding}" == "0" ]] && [[ "${OILFS_NUMA_NUM_CORES-}" == "24" ]] && export NUMA_PHYSBIND="0-11,112-123"
  [[ "${numa_binding}" == "1" ]] && [[ "${OILFS_NUMA_NUM_CORES-}" == "24" ]] && export NUMA_PHYSBIND="56-67,168-179"

  local extra_flags=""

  if [[ $fuse_src_path = "" ]]; then
    extra_flags="${extra_flags} --ws-mount-root"
  fi

  if [[ ${OILFS_USE_LEGACY_SCRIPT+set} && "${OILFS_USE_LEGACY_SCRIPT}" == 1  ]]; then
    /packages/oil.oilfs/scripts/genai_wrapper.sh "$fuse_src" "$fuse_dst" -u "${AI_RM_ATTRIBUTION-}"
  else
    # KEEP THIS HERE UNTIL WE HAVE HAD AT LEAST 1 FULL WEEK OF STABLE WITHOUT ROLLBACK PAST oil.oilfs:663 (see D63648567)
    OILFS_EXTRA_FLAGS_GENAI=$(echo "${OILFS_EXTRA_FLAGS_GENAI-}" | tr @^ ' '=)
    OILFS_EXTRA_FLAGS_PRETRAINING=$(echo "${OILFS_EXTRA_FLAGS_PRETRAINING-}" | tr @^ ' '=)
    /packages/oil.oilfs/oilfs-wrapper --profile="${OILFS_PROFILE-genai}" --log-level debug --user="${AI_RM_ATTRIBUTION-}" ${extra_flags} "$fuse_src" "$fuse_dst"
  fi

  # Clean up env variable after oilfs is mounted
  unset NUMA_BINDING
  unset NUMA_PHYSBIND
}

#######################################
# Mounts airstore with the right setup.
# Globals:
#   ENABLE_AIRSTORE enable airstore (default unset)
#   AIRSTORE_URI allows overriding the oilfs region used for airstore mount.
#######################################
function mount_airstore {
  if [[ -z "${ENABLE_AIRSTORE-}" ]]; then
    echo "Airstore has not been enabled through env ENABLE_AIRSTORE. Skipping mounts."
    return 0
  fi

  local airstore_uri="${AIRSTORE_URI-}"
  if [[ -z "$airstore_uri" ]]; then
    local host
    host="$(hostname)"

    case $host in
      *.pci* )
        airstore_uri="ws://ws.ai.pci0ai/airstore"
        ;;
      *.eag* )
        airstore_uri="ws://ws.ai.eag0genai/airstore"
        ;;
      *.gtn* )
        airstore_uri="ws://ws.ai.gtn0genai/airstore"
        ;;
      *.nha* )
        airstore_uri="ws://ws.ai.nha0genai/airstore"
        ;;
      *.snb* )
        airstore_uri="ws://ws.ai.snb0genai/airstore"
        ;;
      *.nao* )
        airstore_uri="ws://ws.ai.nao0ai/airstore"
        ;;
      * )
        echo -e "\e[31mNo airstore source available based on region of $host, only available in pci, eag, gtn, nha. You can mount a cross-region airstore by passing in the AIRSTORE_URI environment variable\e[0m" 1>&2
        exit 1
        ;;
    esac
  fi

  local mount_dir="${AIRSTORE_LOCAL_MOUNT_ROOT:-/data/users/airstore}"
  if [ ! -d "$mount_dir" ] ; then
    mkdir -p "$mount_dir"
  fi

  echo "WS-Airstore: mount from $airstore_uri to $mount_dir"
  if [[ ${OILFS_USE_LEGACY_SCRIPT+set} && "${OILFS_USE_LEGACY_SCRIPT}" == 1  ]]; then
    /packages/oil.oilfs/scripts/airstore_wrapper.sh "$airstore_uri" "$mount_dir"
  else
    /packages/oil.oilfs/oilfs-wrapper --log-level debug --profile=airstore "$airstore_uri" "$mount_dir" --user "airstore-${AI_RM_ATTRIBUTION-}"
  fi
}


#######################################
# Set up oilfs and airstore mounts if set.
# Globals:
#   DISABLE_OILFS kill switch to skip setting up oilfs and airstore entirely (default unset)
#   AI_RM_ATTRIBUTION should be set by mast for attribution
#
#   FUSE_DST path on host to mount to; defaults to /mnt/wsfuse. For multiple paths, this will
#            be a symlink to the actual path
#
#   For mounting a single path
#   FUSE_SRC which ws path to mount; if unset will default to the region in which
#            the job is running
#   FUSE_SRC_PATH path in ws://ws.ai.<region>0<genai/ai>/<path>; this is a region aware FUSE_SRC
#                 FUSE_SRC takes precedence over FUSE_SRC_PATH if both are set
#
#   For mounting multiple paths
#   FUSE_DST_ROOT  root folder for mounting multiple paths
#   FUSE_SRC_PATHS allow mounting multiple different FUSE_SRCs into /mnt/wsfuse. This will nest
#                  the folders into /mnt/wsfuse instead of mounting directly into /mnt/wsfuse.
#                  Takes precedence over the single path if set
#
#   For mounting root of one or more clusters at /mnt/wsf
#   WARNING: please don't change the mounting location or try to mount other than cluster root
#            as the intention here is to standardise.
#   FUSE_REGIONS comma-separated list of region short names to make available in /mnt/wsf
#                ie FUSE_REGIONS=pci,eag
#######################################
function setup_oilfs_and_airstore {
  if [[ -n "${DISABLE_OILFS-}" ]]; then
    echo "OilFS and Airstore disabled through env DISABLE_OILFS=$DISABLE_OILFS. Skipping mounts."
    return 0
  fi

  local final_fuse_dst="${FUSE_DST:-/mnt/wsfuse}"
  local fuse_dst_root="${FUSE_DST_ROOT:-/mnt/wsf}"

  if [[ -n "${FUSE_REGIONS-}" ]]; then
    # FUSE_REGIONS if specified overrides normal FUSE_SRC_PATH functionality
    # and ignores FUSE_SRC_PATHS. If you use FUSE_REGIONS your backwards-compatible
    # FUSE_DST (default: /mnt/wsfuse) will be a symlink to /mnt/wsf/local/$FUSE_SRC_PATH
    # This will only work if the job is in a region with local WSF. Unlike default
    # functionality the script WILL NOT abort if not local, the symlink creation
    # will just be skipped.
    if [[ "${FUSE_REGIONS}" = "all" ]]; then
      regions_list=(
        "pci"
        "eag"
        "gtn"
        "nha"
        "snb"
        "nao"
      )
    else
      IFS=',' read -r -a regions_list <<< "$FUSE_REGIONS"
    fi

    for region in "${regions_list[@]}"; do
      setup_oilfs_regional "$region"
    done

  elif [[ -n "${FUSE_SRC_PATHS-}" ]]; then
    # FUSE_SRC_PATHS if specified overrides FUSE_SRC_PATH
    IFS=',' read -r -a src_path_list <<< "$FUSE_SRC_PATHS"

    for fuse_src_path in "${src_path_list[@]}"; do
      local fuse_dst_path="${fuse_dst_root}/${fuse_src_path}"
      mount_oilfs "$fuse_src_path" "$fuse_dst_path"
    done

    mkdir -p "$(dirname "$final_fuse_dst")"
    ln -s "${fuse_dst_root}/${src_path_list[0]}" "$final_fuse_dst"
  else
    # default functionality - use FUSE_SRC_PATH, FUSE_DST
    mount_oilfs "${FUSE_SRC_PATH:-genai_fair_llm}" "$final_fuse_dst" "${FUSE_SRC-}"
  fi

  mount_airstore
}

#######################################
# Mount the root directory of every regional AI WSF cluster
# Globals:
#   FUSE_DST_ROOT  root folder for mounting regional WSF roots
#   FUSE_SRC_PATH  path relative to local cluster root to mount at FUSE_DST
#   FUSE_DST       backwards-compatible mount path, defaults to /mnt/wsfuse
function setup_oilfs_regional {
  local region="${1-}"
  local fuse_dst_root="${FUSE_DST_ROOT:-/mnt/wsf}"
  local fuse_src_path="${FUSE_SRC_PATH:-genai_fair_llm}"

  case $region in
    pci|nao)
      mount_oilfs "" "${fuse_dst_root}/${region}" "ws://ws.ai.${region}0ai/"
      ;;
    eag|gtn|nha|snb)
      mount_oilfs "" "${fuse_dst_root}/${region}" "ws://ws.ai.${region}0genai/"
      ;;
    *)
      echo "ERROR: Unrecognized region: ${region}"
      exit 1
      ;;
  esac

  # now create a local symlink, and provide backwards compatibility with /mnt/wsfuse
  if [[ "$REGION_DATACENTER_PREFIX" = "$region" ]]; then
    ln -sT "${fuse_dst_root}/${region}" "${fuse_dst_root}/local"
    ln -sT "${fuse_dst_root}/local/${fuse_src_path}" "${FUSE_DST:-/mnt/wsfuse}"

    # HACK: prime the cache by walking each path component to make the system
    # avoid doing lookup() which currently is not implemented properly with
    # virtual directories
    local path_so_far="${fuse_dst_root}/local"
    IFS='/' read -r -a path_components <<<  "$fuse_src_path"
    for path_component in "${path_components[@]}"
    do
      path_so_far="${path_so_far}/${path_component}"
      if ! ls "$path_so_far" 1>&2
      then
        echo "ERROR: could not ls ${path_so_far}, maybe you don't have permission to access your FUSE_SRC_PATH?" 1>&2
        exit 1
      fi
    done
  fi
}

#######################################
# Mount the supplied Manifold bucket via Manifuse (OILFS-MF)
# Globals:
#   DISABLE_MANIFUSE    kill switch to disable Manifuse setup
#   MANIFUSE_BUCKET     which Manifold bucket to mount; if unset will skip setup
#   MANIFUSE_SRC_PATH   path in manifold://<bucket>/<path>; if unset will mount the root node
#   MANIFUSE_FUSE_DST   path on host to mount to; defaults to /mnt/<bucket>
#   AI_RM_ATTRIBUTION   should be set by mast for attribution
function setup_manifuse {
  if [[ -n "${DISABLE_MANIFUSE-}" ]]; then
    echo "Manifuse disabled through env DISABLE_MANIFUSE=$DISABLE_MANIFUSE. Skipping mounts."
    return 0
  fi

  if [[ -z "${MANIFUSE_BUCKET-}" ]]; then
    echo "Manifold bucket is not set (MANIFUSE_BUCKET is empty), skipping setting up Manifuse."
    return 0
  fi

  local mount_dst="${MANIFUSE_FUSE_DST:-/mnt/$MANIFUSE_BUCKET}"
  local manifold_uri="manifold://${MANIFUSE_BUCKET}/${MANIFUSE_SRC_PATH-}"

  # make the directory if it doesn't already exist
  mkdir -p "$mount_dst"

  # execute oilfs wrapper with profile=manifold
  /packages/oil.oilfs/oilfs-wrapper --profile=manifold --user="${AI_RM_ATTRIBUTION-}" "$manifold_uri" "$mount_dst"
}

#######################################
# Set up ManifoldFS mount if configured.
# Globals:
#   DISABLE_MANIFOLDFS kill switch to skip setting up manifoldfs (default unset)
#   MANIFOLDFS_FUSE_DST path on host to mount to; defaults to /mnt/mffuse
#   MANIFOLDFS_BUCKET which Manifold bucket to mount; if unset will skip setup
# NOTE: ManifoldFS will be deprecated shortly in favor of Manifuse (see above)
#######################################
function setup_manifoldfs {
  if [[ -n "${DISABLE_MANIFOLDFS-}" ]]; then
    echo "ManifoldFS disabled through env DISABLE_MANIFOLDFS=$DISABLE_MANIFOLDFS. Skipping mounts."
    return 0
  fi

  if [[ -z "${MANIFOLDFS_BUCKET-}" ]]; then
    echo "Manifold bucket is not set (MANIFOLDFS_BUCKET is empty), skipping setting up ManifoldFS"
    return 0
  fi

  MANIFOLDFS_FUSE_DST="${MANIFOLDFS_FUSE_DST:-/mnt/mffuse}"
  mkdir -p "${MANIFOLDFS_FUSE_DST}"

  MANIFOLDFS_BINARY=${MANIFOLDFS_BINARY:-"/packages/manifold.manifoldfs/manifoldfs"}

  "${MANIFOLDFS_BINARY}" "manifold.blobstore" "${MANIFOLDFS_BUCKET}" "${MANIFOLDFS_FUSE_DST}"
}

setup_oilfs_and_airstore
setup_manifoldfs
setup_manifuse
