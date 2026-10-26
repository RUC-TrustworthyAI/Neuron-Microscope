# Contributing

Start with [the method map](docs/method.md) and [development checks](docs/development.md). Keep changes to packaging and changes to measurement methods separate.

For a measurement change, explain the unit, input prefix, scoring position, dtype, sampling rule and effect on saved results. Use tiny synthetic or randomly initialized model fixtures for software checks. Never silently alter raw responses or patch generated reports by hand.

For a report change, update the renderer and regenerate from saved inputs. Preserve the requested section structure; do not add metadata panels or a Capture setting table.

Do not commit weights, activation archives, credentials, full API logs or private corpus examples. Keep local settings in `settings.local.json`. Record failures and limitations honestly; passing software tests does not validate a neuron interpretation.
