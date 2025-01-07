# summary.py

import logging

logger = logging.getLogger(__name__)


class Summary:
    def __init__(self):
        self.results = []

    def add_result(self, set_name, resolution, error):
        self.results.append({
            'set_name': set_name,
            'resolution': resolution,
            'error': error
        })

    def summarize(self):
        if not self.results:
            logger.info("No results to summarize.")
            return

        logger.info("==== Experiment Summary ====")
        for r in self.results:
            logger.info(
                f"Set: {r['set_name']} | "
                f"Resolution: {r['resolution']} | "
                f"Error: {r['error']:.4f}"
            )
        logger.info("==== End of Summary ====")
