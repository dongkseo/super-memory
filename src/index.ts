#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { graph, server } from "./server.js";

async function main() {
  await graph.load();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("[fatal]", err);
  process.exit(1);
});
