TAR_DIR=000172

if [ -d traj_segs/$TAR_DIR ]; then
  tar -cf traj_segs/$TAR_DIR.tar traj_segs/$TAR_DIR
  rm -rf traj_segs/$TAR_DIR
fi
