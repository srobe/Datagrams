#!/bin/csh -f

# Set up environment
# ==================

  set dattim = $1
  set config = $2

  setenv GRAM_IMG_PATH `get_param.py $dattim $config GRAM_IMG_PATH`
  setenv GRAM_PUBLISH_PATH `get_param.py $dattim $config GRAM_PUBLISH_PATH`
  setenv STATION_PATH `get_param.py $dattim $config STATION_PATH`

  /usr/bin/rsync -av -e "ssh -i /home/dao_ops/.ssh/id_dataport" \
                 $GRAM_IMG_PATH/ dataportal-key:$GRAM_PUBLISH_PATH/gram/ \
                 --delete --ignore-times --exclude 'meteo_*.png' \
                 --exclude 'du*.png' --exclude 'tot*.png' --exclude 'oc*.png' \
                 --exclude 'bc*.png' --exclude 'ss*.png' --exclude 'su*.png' \
                 --exclude 'ni*.png' --exclude 'co*.png' --exclude=menus

  if ($status != 0) then
    exit 1
  endif

  /usr/bin/rsync -av -e "ssh -i /home/dao_ops/.ssh/id_dataport" \
                 $STATION_PATH/ \
                 dataportal-key:$GRAM_PUBLISH_PATH/gram/menus/ \
                 --delete --ignore-times

  if ($status != 0) then
    exit 2
  endif

  ssh -i /home/dao_ops/.ssh/id_dataport dataportal-key \
         "chmod -R 755 $GRAM_PUBLISH_PATH/gram"

  if ($status != 0) then
    exit 3
  endif

exit 0
