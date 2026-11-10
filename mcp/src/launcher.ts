import { VERSION } from "./version.js";

export interface EngineCommand {
  command: string;
  args: string[];
}

/**
 * Decide how to start the engine sidecar.
 *
 * Dev override: KIBBLE_ENGINE_CMD runs the working-tree engine so you never
 * need to publish to test. Otherwise run the exact matching engine version
 * through uvx. (Whitespace splitting is deliberate and simple: the override
 * is a developer convenience, not a shell.)
 *
 * Still to build: free-port handshake, session token, health polling,
 * startup timeout with captured stderr, and orphan-safe shutdown.
 */
export function resolveEngineCommand(
  env: Record<string, string | undefined> = process.env,
): EngineCommand {
  const override = env["KIBBLE_ENGINE_CMD"]?.trim();
  if (override) {
    const [command, ...args] = override.split(/\s+/);
    if (command) return { command, args };
  }
  return {
    command: "uvx",
    args: ["--from", `kibble-engine==${VERSION}`, "kibble-engine", "serve"],
  };
}
