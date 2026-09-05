# Provider pricing

`provider_pricing.json` is the local, versioned pricing source loaded at startup.
It intentionally contains no prices: provider tariffs and account contracts change.

Add immutable entries using the provider/model names selected in FermaYT:

```json
{
  "prices": [
    {
      "provider": "qwen",
      "model": "qwen-image-3.0",
      "operation": "NEW_IMAGE",
      "pricing_unit": "PER_IMAGE",
      "price": 0.0,
      "currency": "USD",
      "version": "replace-with-provider-price-version",
      "effective_from": "2026-01-01T00:00:00+00:00"
    }
  ]
}
```

Replace `price`, `version`, and `effective_from` with the current official or
contracted tariff. Do not use the example zero as a real tariff. Supported units
are `PER_IMAGE`, `PER_CHARACTER`, and `PER_REQUEST`. Current operation names are
`PLANNING`, `NEW_IMAGE`, `REFERENCE_GENERATION`, `EDIT`, `VISUAL_QA`, and `TTS`.

Set `FERMAYT_PRICING_FILE` to load a different JSON file. Existing versions are
never overwritten; add a new version/effective date when a tariff changes.

When a Project budget is enabled, every provider/model/operation used by that
Project must have a price in the same currency as the budget. Missing or mixed
currency pricing pauses generation before the request; it is never treated as
free.
