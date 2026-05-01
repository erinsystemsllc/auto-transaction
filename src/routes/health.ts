import { NextFunction, Request, Response, Router } from 'express';

const router = Router();

router.get('/', (req: Request, res: Response, next: NextFunction) => {
  res.send(`Everything OK! Running in ${process.env.NODE_ENV}`);
  next();
});

export default router;
