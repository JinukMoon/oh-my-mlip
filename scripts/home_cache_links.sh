#!/usr/bin/env bash
# home_cache_links.sh -- point the caches some frameworks hard-code under $HOME
# at $OMM_SHARED_CACHE_ROOT/home/..., with symlinks.
#
# HPC home directories are small (often a few GB or a few tens of thousands of
# files). These frameworks write their weights to a fixed path under $HOME,
# whatever XDG_CACHE_HOME or the other cache variables say:
#   ~/.cache/sevennet (SevenNet)  ~/.cache/matris (MatRIS)  ~/.cache/tace (TACE)
#   ~/.cache/nequix (Nequix)      ~/.cache/eqnorm (Eqnorm)  ~/.cache/grace (GRACE)
#   ~/.cache/deepmd (DeePMD/DPA)  ~/.local/mattersim (MatterSim)
# For each one that does not exist yet, this creates the target directory under
# $OMM_SHARED_CACHE_ROOT/home and a symlink at the $HOME path. An existing
# directory is left alone and listed (move it yourself if you want it out of
# $HOME). An existing symlink is kept, except that a link to this root whose
# directory was deleted gets it back, and a dangling link (into a root that no
# longer exists) is pointed here. Does nothing without OMM_SHARED_CACHE_ROOT.
# The links stay in $HOME after the hub or cache root is deleted.
# install.sh runs it before downloading weights.
set -u
root="${OMM_SHARED_CACHE_ROOT:-}"
[ -n "$root" ] || exit 0
kept=()
made=()
recreated=()
repointed=()
for rel in .cache/sevennet .cache/matris .cache/tace .cache/nequix .cache/eqnorm .cache/grace \
           .cache/deepmd .local/mattersim; do
  link="$HOME/$rel"
  target="$root/home/$rel"
  if [ -L "$link" ]; then
    dest="$(readlink "$link")"
    if [ "$dest" = "$target" ]; then
      # ours: its directory may be gone (a deleted hub or cache root)
      [ -d "$target" ] || { mkdir -p "$target" && recreated+=("$target"); }
    elif [ ! -e "$link" ]; then
      # dangling, into some other root (an earlier, deleted install): point it here
      rm -f "$link" && mkdir -p "$target" && ln -s "$target" "$link" && repointed+=("$link")
    fi
    continue
  elif [ -e "$link" ]; then
    kept+=("$link")
    continue
  fi
  mkdir -p "$target" "$(dirname "$link")" && ln -s "$target" "$link" && made+=("$link")
done
if [ "${#made[@]}" -gt 0 ]; then
  echo "  home caches -> $root/home: ${made[*]}"
fi
if [ "${#recreated[@]}" -gt 0 ]; then
  echo "  home cache link targets re-created (they had been deleted): ${recreated[*]}"
fi
if [ "${#repointed[@]}" -gt 0 ]; then
  echo "  home cache links that pointed at a deleted directory now point into $root/home: ${repointed[*]}"
fi
if [ "${#kept[@]}" -gt 0 ]; then
  echo "  note: these caches already exist in \$HOME and stay there (move them under $root/home and" >&2
  echo "  leave a symlink if your home quota is small): ${kept[*]}" >&2
fi
exit 0
