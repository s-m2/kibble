import { describe, expect, it } from "vitest";
import { resolveEngineCommand } from "../src/launcher.js";
import { VERSION } from "../src/version.js";

describe("resolveEngineCommand", () => {
  it("defaults to uvx with the exact matching engine version", () => {
    expect(resolveEngineCommand({})).toEqual({
      command: "uvx",
      args: ["--from", `kibble-engine==${VERSION}`, "kibble-engine", "serve"],
    });
  });

  it("honours the dev override", () => {
    const cmd = resolveEngineCommand({
      KIBBLE_ENGINE_CMD: "uv run --project ../engine kibble-engine serve",
    });
    expect(cmd.command).toBe("uv");
    expect(cmd.args).toEqual(["run", "--project", "../engine", "kibble-engine", "serve"]);
  });

  it("ignores a blank override", () => {
    expect(resolveEngineCommand({ KIBBLE_ENGINE_CMD: "   " }).command).toBe("uvx");
  });
});
