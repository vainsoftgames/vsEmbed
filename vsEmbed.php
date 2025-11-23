<?php
	class vsEmbed {
		public string $embed_url = "http://{PATH TO EMBED SERVER}/embed_text";
		public string $search_url = "http://[PATH TO EMBED DATABASE SERVER}/searchEntry";
		public string $collection = "{COLLECTION NAME}";

		/**
		 * Generic POST wrapper
		 */
		private function callAPI(string $url, array $payload) : array {
			$ch = curl_init($url);

			curl_setopt_array($ch, [
				CURLOPT_RETURNTRANSFER => true,
				CURLOPT_POST           => true,
				CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
				CURLOPT_POSTFIELDS     => json_encode($payload),
				CURLOPT_CONNECTTIMEOUT => 3,
				CURLOPT_TIMEOUT        => 10
			]);

			$out = curl_exec($ch);
			$err = curl_error($ch);

			curl_close($ch);

			if ($err) {
				throw new Exception("API Error: " . $err);
			}

			$json = json_decode($out, true);
			if (!$json) {
				throw new Exception("Invalid JSON response from: $url\nResponse: $out");
			}

			return $json;
		}

		/**
		 * Generate embedding(s)
		 * Accepts string or array of strings.
		 */
		public function embed(string|array $text) : array {
			if (is_string($text)) {
				$text = [$text];     // Always send array to embedder
			}

			$payload = [
				"texts" => $text,
				"normalize" => true
			];

			$response = $this->callAPI($this->embed_url, $payload);

			// Assuming your embed server returns {"embeddings": [[...]]}
			if (!isset($response["embeddings"])) {
				throw new Exception("Embed server missing 'embeddings' key");
			}

			return $response["embeddings"];
		}

		/**
		 * Search Milvus using embedded query
		 */
		public function search(string $text, bool|array|string $fields, int $limit = 10) : vsEmbedResult  {
			$vec = $this->embed($text)[0]; // take first vector

			$payload = [
				"collection_name" => $this->collection,
				"vector" => $vec,
				"top_k" => $limit,
				"metric" => "COSINE"
			];
			if($fields){
				$payload['output_fields'] = (is_array($fields) ? $fields : [$fields]);
			}

			$response = $this->callAPI($this->search_url, $payload);
			return new vsEmbedResult($response);
		}
	}

	class vsEmbedResult {
		private array $results = [];
		private int $position = 0;

		public function __construct(array $data){
			$this->results = isset($data['results']) && is_array($data['results'])
				? $data['results']
				: [];
		}

		public function num_rows() : int {
			return count($this->results);
		}

		/**
		 * Fetch next row as assoc array (MySQL-style).
		 * Returns null when no more rows.
		 *
		 * Shape:
		 * [
		 *   'id'    => mixed,
		 *   'score' => float,
		 *   ...payload fields...
		 * ]
		 */
		public function fetch_assoc() : ?array {
			if ($this->position >= $this->num_rows()) {
				return null;
			}

			$raw = $this->results[$this->position++];
			$row = [];

			// Try to map id
			if (isset($raw['id'])) {
				$row['id'] = $raw['id'];
			} 
			elseif (isset($raw['primary_key'])) {
				$row['id'] = $raw['primary_key'];
			}

			// Try to map similarity/score (distance vs score)
			if (isset($raw['score'])) {
				$row['score'] = $raw['score'];
			}
			elseif (isset($raw['distance'])) {
				$row['score'] = $raw['distance'];
			}

			// Flatten payload if present
			if (isset($raw['payload']) && is_array($raw['payload'])) {
				foreach ($raw['payload'] as $k => $v) {
					// Don’t overwrite id/score if already set
					if (!array_key_exists($k, $row)) {
						$row[$k] = $v;
					}
				}
			}

			// If your API already returns flat docs in `result` or `entity`, you can also merge that in here.
			return $row;
		}
		

		/**
		 * Reset internal pointer if you want to iterate again
		 */
		public function rewind() : void {
			$this->position = 0;
		}

		/**
		 * Convenience: fetch all rows
		 */
		public function fetch_all() : array {
			$this->rewind();
			$out = [];
			while ($row = $this->fetch_assoc()) {
				$out[] = $row;
			}
			return $out;
		}
	}
?>
