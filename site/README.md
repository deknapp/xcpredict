# site/

A static page for browsing the model's predictions against what actually
happened. No server: every prediction here is about a race that has already
been run, so there is nothing to compute per visitor.

## Rebuilding

```bash
xcpredict train      # fit the ranker, write data/model.json
xcpredict export     # freeze predictions into site/data/
```

Then open it locally:

```bash
cd site && python3 -m http.server 8000
```

## Not published

This is deliberately not deployed anywhere yet. `site/data/` is committed so
the page works for anyone who clones the repo, and turning on GitHub Pages
would be a separate, deliberate decision.
