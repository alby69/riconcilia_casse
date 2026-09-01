FROM nginx:alpine AS runtime

COPY web/ /usr/share/nginx/html/
COPY app/manifest.json /usr/share/nginx/html/manifest.json
COPY app/sw.js /usr/share/nginx/html/sw.js

COPY docker/nginx.conf /etc/nginx/templates/default.conf.template

ENV PORT=80
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/ || exit 1

CMD ["sh", "-c", "envsubst '${PORT}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
