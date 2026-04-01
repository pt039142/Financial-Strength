import { ReactNode } from "react";
import Navbar from "@/components/Navbar";
import SiteFooter from "@/components/SiteFooter";

interface SiteFrameProps {
  children: ReactNode;
}

const SiteFrame = ({ children }: SiteFrameProps) => {
  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-10" />
      <div className="pointer-events-none absolute left-1/2 top-0 h-[36rem] w-[36rem] -translate-x-1/2 rounded-full bg-primary/10 blur-[140px]" />
      <Navbar />
      <main className="relative z-10">{children}</main>
      <SiteFooter />
    </div>
  );
};

export default SiteFrame;
