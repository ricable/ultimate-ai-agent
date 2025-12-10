import { Request, Response, NextFunction } from "express";
import Module from "node:module";
import { streamOpenAIResponse } from "../utils/stream";
import { log } from "../utils/log";
import { PLUGINS_DIR } from "../constants";
import path from "node:path";
import { access } from "node:fs/promises";
import { OpenAI } from "openai";
import { createClient } from "../utils";

// @ts-ignore
const originalLoad = Module._load;
// @ts-ignore
Module._load = function (request, parent, isMain) {
  if (request === "claude-code-router") {
    return {
      streamOpenAIResponse,
      log,
      OpenAI,
      createClient,
    };
  }
  return originalLoad.call(this, request, parent, isMain);
};

export const rewriteBody = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (!req.config.usePlugins) {
    return next();
  }
  for (const plugin of req.config.usePlugins) {
    const pluginPath = path.join(PLUGINS_DIR, `${plugin.trim()}.js`);
    try {
      await access(pluginPath);
      const rewritePlugin = require(pluginPath);
      await rewritePlugin(req, res);
    } catch (e) {
      console.error(e);
    }
  }
  next();
};
