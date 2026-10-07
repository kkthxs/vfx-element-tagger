# Security and Privacy

Browser-controlled jobs can read user-selected local folders and write the configured
catalog/cache. Keep the server on loopback and never expose it through a tunnel. The
folder browser lists local directories; it is not a cloud upload. Model manifests and
command wrappers are trusted local executable configuration, not web-editable inputs.
Jobs share a cooperative writer lease with the current ingest/analyzer CLI; other
maintenance tools must run offline. Automatic backups and job logs contain local paths
and must stay out of public source releases.

VFX Element Tagger is a local workstation application, not a hardened hosted service.
Keep the default bind address `127.0.0.1`. Do not port-forward it or expose it directly
to the public internet.

The built-in HTTP service refuses non-loopback binding, checks Host/Origin and cross-site
request metadata, bounds JSON request bodies and denies framing. It has no login,
authorization or TLS. Artist names and browser-generated rating IDs are not authenticated
identities. LAN, reverse-proxy, tunnel and public hosting are unsupported by v1.

The default model workflow runs locally after downloading weights. Downloads contact
Hugging Face. Source paths, captions, predictions, reviewer notes and artist rating names
are stored in catalogs and may be exposed to anyone who can reach the service. Browser
local storage retains the rating identity. Custom command backends execute local programs
and may have their own network behavior; trust manifests as executable configuration.

Only ingest trusted media using maintained FFmpeg, image decoders and Python packages.
Do not load untrusted model snapshots, remote model code or manifests. This application
does not sandbox decoders, model loaders or custom commands.

Never attach private footage, model weights, access tokens, `.env` files, live catalogs,
or unsanitized logs to public issues. Ignoring a file does not remove it from Git history.
Review source archives and all commits before first publication.

Use `scripts/export_public_release.py` to create a clean allowlisted source tree without
the working repository's history, private reports, models, footage, catalogs or article
screenshots. Publish that checked export, not an unreviewed push of development history.

Report suspected vulnerabilities through
[GitHub's private vulnerability-reporting channel](https://github.com/kkthxs/vfx-element-tagger/security/advisories/new).
Private reporting is enabled for this repository. Do not post exploit details, private
media, credentials or unsanitized catalogs/logs in a public issue.
