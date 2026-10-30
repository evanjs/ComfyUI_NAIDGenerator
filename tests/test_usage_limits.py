import unittest

from usage_limits import (
    format_opus_usage,
    get_opus_usage,
    get_subscription,
    opus_usage_result,
    should_block_v5_generation,
)


SAMPLE_SUBSCRIPTION = {
    "tier": 3,
    "active": True,
    "usage": {
        "percent": 93,
        "purchasedPercent": 0,
        "isNegative": False,
        "timeUntilNextPercent": 6048,
    },
}


class OpusUsageTests(unittest.TestCase):
    def test_subscription_endpoint_payload_is_accepted_without_wrapper(self):
        self.assertEqual(get_subscription(SAMPLE_SUBSCRIPTION), SAMPLE_SUBSCRIPTION)

    def test_sample_usage_reports_percent_and_estimated_images_and_refill(self):
        usage = get_opus_usage({"subscription": SAMPLE_SUBSCRIPTION})
        self.assertEqual(usage["percent"], 93)
        self.assertEqual(
            format_opus_usage(usage),
            "93% remaining (~1609 images)\nCurrently refills at 14.3% per day (~247 images).\nImage counts are approximate, based on the supplied standard-resolution sample.",
        )

    def test_tracker_returns_visible_text_and_standard_node_outputs(self):
        result = opus_usage_result({"subscription": SAMPLE_SUBSCRIPTION})
        self.assertEqual(result["result"], (93, format_opus_usage(get_opus_usage({"subscription": SAMPLE_SUBSCRIPTION}))))
        self.assertEqual(result["ui"]["text"], [result["result"][1]])

    def test_zero_usage_blocks_only_active_opus_v5(self):
        subscription = {"tier": 3, "active": True, "usage": {"percent": 0}}
        self.assertTrue(should_block_v5_generation(subscription, "nai-diffusion-5-full"))
        self.assertTrue(should_block_v5_generation(subscription, "nai-diffusion-5-curated"))
        self.assertFalse(should_block_v5_generation(subscription, "nai-diffusion-4-5-full"))
        self.assertFalse(should_block_v5_generation({**subscription, "tier": 2}, "nai-diffusion-5-full"))
        self.assertFalse(should_block_v5_generation({**subscription, "active": False}, "nai-diffusion-5-full"))

    def test_usage_percent_is_clamped_to_valid_range(self):
        self.assertEqual(get_opus_usage({"subscription": {"usage": {"percent": 105}}})["percent"], 100)
        self.assertEqual(get_opus_usage({"subscription": {"usage": {"percent": -4}}})["percent"], 0)


if __name__ == "__main__":
    unittest.main()
