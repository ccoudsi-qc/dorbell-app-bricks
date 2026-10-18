# dorbell-app-bricks

Arduino **App Lab** examples for the **Dragonwing** family — the prototype-side of
a face-recognition doorbell that runs InsightFace `buffalo_sc` on a Qualcomm NPU.

This repo follows the layout of
[`arduino/app-bricks-examples`](https://github.com/arduino/app-bricks-examples):
example apps live under `inspirational/<platform>/<app>/`, shared services live at
the repo root, and `examples.json` is the manifest App Lab reads.

## The one example: Dragonwing Doorbell (Ventuno Q)

[`inspirational/platform_ventunoq/dragonwing-doorbell/`](inspirational/platform_ventunoq/dragonwing-doorbell/README.md)

An autonomous face-recognition doorbell for the **Ventuno Q** (`arduino,monza` /
QCS8300, HTP v75), packaged as an App Lab app:

- a custom app-local **`arduino:face_recognition`** brick — detection + 512-d
  ArcFace embeddings on the NPU, served by a companion container (the
  [`runner/`](runner/README.md));
- the stock **`arduino:web_ui`** brick — the full doorbell UI, streamed into a
  webview at `:7000`, with key events coming back over the same Socket.IO channel;
- QR-triggered enrollment and a 2-minute in-memory TTL store, reused **verbatim**
  from the production app.

```
dorbell-app-bricks/
├── inspirational/
│   └── platform_ventunoq/
│       └── dragonwing-doorbell/   # the App Lab app (app.yaml, python/, assets/)
├── runner/                        # the NPU inference container behind the brick
├── examples.json                  # App Lab manifest
├── install-bricks.md              # fresh Ubuntu + App Lab board → running doorbell
└── LICENSE.txt                    # MPL-2.0
```

## Prototype → production

This repo is the **prototype** half of a larger story. The *same* doorbell UI and
state machine also run as a bare-metal / containerized app on the **IQ-8275**
(NPU reached via a CDI (Container Device Interface) file + bundled `qai-appbuilder`).
The App Lab app here imports that production code **unchanged** —
`python/doorbell_core.py` is a byte-for-byte snapshot of the production
`doorbell.py` — and swaps only the two I/O ends (inference → the brick/runner;
display → the web_ui webview).

The production path and the narrative that connects the two live in
**[munoz0raul/dorbell](https://github.com/munoz0raul/dorbell)**.

## Getting started

New board? **[`install-bricks.md`](install-bricks.md)** takes a fresh Ubuntu +
App Lab Ventuno Q to a running doorbell step by step. Already set up? See the app's
own [README](inspirational/platform_ventunoq/dragonwing-doorbell/README.md) for
`run_applab.sh` and the controls.

## License

[MPL-2.0](LICENSE.txt). Portions adapted from
[`arduino/app-bricks-examples`](https://github.com/arduino/app-bricks-examples).
