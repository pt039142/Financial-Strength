import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Link } from "react-router-dom";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="max-w-lg rounded-3xl border border-white/10 bg-card/80 p-10 text-center shadow-[0_20px_80px_-35px_rgba(0,0,0,0.65)] backdrop-blur">
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-primary">Autonomiq.AI</p>
        <h1 className="mb-4 mt-4 text-4xl font-heading font-bold">404</h1>
        <p className="mb-8 text-lg text-muted-foreground">Oops! That page does not exist yet.</p>
        <Link to="/" className="inline-flex rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90">
          Return to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
