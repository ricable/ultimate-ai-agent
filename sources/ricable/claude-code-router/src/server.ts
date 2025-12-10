import express, { RequestHandler } from "express";

interface Server {
  app: express.Application;
  useMiddleware: (middleware: RequestHandler) => void;
  start: () => void;
}

export const createServer = async (port: number): Promise<Server> => {
  const app = express();
  app.use(express.json({ limit: "500mb" }));
  return {
    app,
    useMiddleware: (middleware: RequestHandler) => {
      app.use("/v1/messages", middleware);
    },
    start: () => {
      app.listen(port, () => {
        console.log(`Server is running on port ${port}`);
      });
    },
  };
};
