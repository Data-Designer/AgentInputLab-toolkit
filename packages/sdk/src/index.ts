/** No automatic retries: one request means at most one submitted attempt. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export interface Budget {
  max_model_calls?: number; max_retrieval_calls?: number; max_tool_calls?: number;
  max_output_tokens?: number; max_context_tokens?: number | null; max_cost_usd?: number | null;
}
export interface State {
  query: string; task_id?: string; instruction?: string;
  context?: { text: string; block_id?: string; source?: string; score?: number | null; metadata?: Record<string, Json> }[];
  messages?: { role: string; content: string; name?: string | null }[];
  tool_names?: string[]; metadata?: Record<string, Json>; budget?: Budget;
}
export interface InterventionRequest { release_id: string; state: State; seed?: number; }
export interface InterventionResult {
  record: { release_id: string; input_state_hash: string; output_state_hash: string;
    patch: Record<string, Json>; status: string; usage: Record<string, Json>; [key: string]: Json; };
  state: State; scored: false; benchmark_score: null;
}
export interface Catalog {
  methods: Record<string, Json>[]; benchmarks: Record<string, Json>[];
  configured_releases: string[]; note: string;
}
export class AgentInputLabError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message); this.name = "AgentInputLabError";
  }
}
export interface ClientOptions {
  baseUrl?: string; token: string; timeoutMs?: number; fetch?: typeof globalThis.fetch;
}
function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function validData(path: string, data: unknown): boolean {
  if (!object(data)) return false;
  if (path === '/v1/health') return data.status === 'ok' && data.protocol === 'agentinputlab.v1';
  if (path === '/v1/catalog') return Array.isArray(data.methods) && data.methods.every(object)
    && Array.isArray(data.benchmarks) && data.benchmarks.every(object)
    && Array.isArray(data.configured_releases) && data.configured_releases.every(x => typeof x === 'string')
    && typeof data.note === 'string';
  const record = data.record;
  return data.scored === false && data.benchmark_score === null
    && object(data.state) && typeof data.state.query === 'string' && object(record)
    && ['release_id', 'input_state_hash', 'output_state_hash', 'status'].every(k => typeof record[k] === 'string')
    && object(record.patch) && object(record.usage);
}
export class AgentInputLab {
  readonly #base: string;
  readonly #token: string;
  readonly #timeout: number;
  readonly #fetch: typeof globalThis.fetch;
  constructor(options: ClientOptions) {
    const url = new URL(options.baseUrl ?? "http://127.0.0.1:8765");
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/')
      throw new TypeError('baseUrl must be an HTTP(S) origin without credentials, path, query or fragment');
    if (url.protocol === 'http:' && !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname))
      throw new TypeError('Remote endpoints require HTTPS');
    if (typeof options.token !== 'string' || !/^[\x21-\x7e]+$/.test(options.token))
      throw new TypeError('A printable ASCII token without whitespace is required');
    this.#timeout = options.timeoutMs ?? 120_000;
    if (!Number.isSafeInteger(this.#timeout) || this.#timeout < 1 || this.#timeout > 2147483647)
      throw new TypeError('timeoutMs must be an integer between 1 and 2147483647');
    this.#base = url.origin; this.#token = options.token;
    this.#fetch = options.fetch ?? globalThis.fetch;
    if (typeof this.#fetch !== 'function') throw new TypeError('A fetch implementation is required');
  }
  async #request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    signal?.throwIfAborted();
    const encoded = body === undefined ? undefined : JSON.stringify(body, (_key, value) => {
      if (typeof value === 'number' && !Number.isFinite(value))
        throw new TypeError('Request numbers must be finite; budgets cannot be NaN or Infinity');
      return value;
    });
    if (encoded !== undefined && new TextEncoder().encode(encoded).byteLength > 1024 * 1024)
      throw new TypeError('Request body exceeds the server limit of 1 MiB');
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => controller.abort(new Error('Request timed out; server execution may still be running. Do not automatically retry.')), this.#timeout);
    try {
      const response = await this.#fetch(this.#base + path, {
        method: body === undefined ? 'GET' : 'POST', redirect: 'error',
        headers: { Authorization: `Bearer ${this.#token}`, 'Content-Type': 'application/json' },
        ...(encoded === undefined ? {} : { body: encoded }), signal: controller.signal,
      });
      let payload: unknown;
      try { payload = await response.json(); }
      catch { throw new AgentInputLabError(response.status, 'invalid_response', 'Server did not return JSON'); }
      if (!object(payload)) throw new AgentInputLabError(response.status, 'invalid_response', 'Invalid response envelope');
      const obj = payload as { data?: T; error?: { code?: unknown; message?: unknown } };
      if (!response.ok) throw new AgentInputLabError(response.status,
        typeof obj.error?.code === 'string' ? obj.error.code : 'http_error',
        typeof obj.error?.message === 'string' ? obj.error.message : 'Request failed');
      if (!('data' in obj)) throw new AgentInputLabError(response.status, 'invalid_response', 'Missing data envelope');
      if (!validData(path, obj.data)) throw new AgentInputLabError(response.status, 'invalid_response', 'Response does not match the requested endpoint contract');
      return obj.data as T;
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
  }
  health(signal?: AbortSignal) { return this.#request<{status: string; protocol: string}>('/v1/health', undefined, signal); }
  catalog(signal?: AbortSignal) { return this.#request<Catalog>('/v1/catalog', undefined, signal); }
  intervene(request: InterventionRequest, signal?: AbortSignal) {
    return this.#request<InterventionResult>('/v1/interventions', request, signal);
  }
}
