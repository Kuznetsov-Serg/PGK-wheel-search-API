#!/bin/sh

for prestart in ${APP_DIR}/prestart*; do
    if [ -f $prestart ]; then
        echo "Running script $prestart"
        /bin/sh $prestart
    fi
done
exec "$@"
