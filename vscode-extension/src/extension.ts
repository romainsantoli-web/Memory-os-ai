import * as vscode from "vscode";
import { exec } from "child_process";
import { promisify } from "util";
import * as path from "path";
import * as fs from "fs";

const execAsync = promisify(exec);

export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel("Memory OS AI");

  // Auto-setup MCP config on activation
  const config = vscode.workspace.getConfiguration("memory-os-ai");
  if (config.get<boolean>("autoStart")) {
    ensureMcpConfig(outputChannel);
  }

  // Command: Setup
  context.subscriptions.push(
    vscode.commands.registerCommand("memory-os-ai.setup", async () => {
      await ensureMcpConfig(outputChannel);
      vscode.window.showInformationMessage(
        "Memory OS AI: MCP config written to .vscode/mcp.json"
      );
    })
  );

  // Command: Status
  context.subscriptions.push(
    vscode.commands.registerCommand("memory-os-ai.status", async () => {
      try {
        const { stdout } = await execAsync("memory-os-ai setup status");
        outputChannel.clear();
        outputChannel.appendLine(stdout);
        outputChannel.show();
      } catch {
        vscode.window.showErrorMessage(
          "Memory OS AI not installed. Run: pip install memory-os-ai"
        );
      }
    })
  );

  // Command: Search memory
  context.subscriptions.push(
    vscode.commands.registerCommand("memory-os-ai.search", async () => {
      const query = await vscode.window.showInputBox({
        prompt: "Search your memory",
        placeHolder: "e.g., authentication flow, database schema...",
      });
      if (!query) return;

      outputChannel.clear();
      outputChannel.appendLine(`Searching memory for: "${query}"...`);
      outputChannel.show();

      try {
        const { stdout } = await execAsync(
          `echo '{"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"${query.replace(/"/g, '\\"')}"}}}' | memory-os-ai`
        );
        outputChannel.appendLine(stdout);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Error: ${msg}`);
      }
    })
  );

  // Command: Ingest folder
  context.subscriptions.push(
    vscode.commands.registerCommand("memory-os-ai.ingest", async () => {
      const folders = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectFiles: false,
        canSelectMany: false,
        openLabel: "Ingest into memory",
      });
      if (!folders || folders.length === 0) return;

      const folderPath = folders[0].fsPath;
      outputChannel.clear();
      outputChannel.appendLine(`Ingesting: ${folderPath}...`);
      outputChannel.show();

      try {
        const { stdout } = await execAsync(
          `echo '{"method":"tools/call","params":{"name":"memory_ingest","arguments":{"path":"${folderPath.replace(/"/g, '\\"')}"}}}' | memory-os-ai`
        );
        outputChannel.appendLine(stdout);
        vscode.window.showInformationMessage(
          `Memory OS AI: Ingested ${folderPath}`
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        outputChannel.appendLine(`Error: ${msg}`);
      }
    })
  );

  outputChannel.appendLine("Memory OS AI extension activated");
}

async function ensureMcpConfig(outputChannel: vscode.OutputChannel) {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) return;

  const wsRoot = workspaceFolders[0].uri.fsPath;
  const vscodePath = path.join(wsRoot, ".vscode");
  const mcpJsonPath = path.join(vscodePath, "mcp.json");

  // Read existing or create new
  let mcpConfig: Record<string, unknown> = {};
  try {
    const content = fs.readFileSync(mcpJsonPath, "utf-8");
    mcpConfig = JSON.parse(content);
  } catch {
    // File doesn't exist — will create
  }

  const servers = (mcpConfig.servers as Record<string, unknown>) || {};

  // Only add if not already configured
  if (!servers["memory-os-ai"]) {
    const config = vscode.workspace.getConfiguration("memory-os-ai");
    const cacheDir = config.get<string>("cacheDir") || "";
    const model = config.get<string>("model") || "all-MiniLM-L6-v2";

    const env: Record<string, string> = {
      MEMORY_MODEL: model,
    };
    if (cacheDir) {
      env.MEMORY_CACHE_DIR = cacheDir;
    }

    servers["memory-os-ai"] = {
      command: "memory-os-ai",
      args: [],
      env,
    };

    mcpConfig.servers = servers;

    if (!fs.existsSync(vscodePath)) {
      fs.mkdirSync(vscodePath, { recursive: true });
    }
    fs.writeFileSync(mcpJsonPath, JSON.stringify(mcpConfig, null, 2) + "\n");
    outputChannel.appendLine(`Wrote MCP config to ${mcpJsonPath}`);
  }
}

export function deactivate() {
  // Cleanup if needed
}
