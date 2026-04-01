import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";

type Metric = {
  value: string;
  label: string;
};

type Highlight = {
  title: string;
  description: string;
};

type SectionBlock = {
  eyebrow: string;
  title: string;
  description: string;
  bullets: string[];
};

interface SectionedPageProps {
  eyebrow: string;
  title: string;
  lead: string;
  summary: string;
  primaryCta: {
    label: string;
    to: string;
  };
  secondaryCta?: {
    label: string;
    to: string;
  };
  metrics: Metric[];
  highlights: Highlight[];
  sections: SectionBlock[];
}

const SectionedPage = ({
  eyebrow,
  title,
  lead,
  summary,
  primaryCta,
  secondaryCta,
  metrics,
  highlights,
  sections,
}: SectionedPageProps) => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="mx-auto max-w-4xl text-center"
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              {eyebrow}
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold leading-tight md:text-5xl lg:text-6xl">
              {title}
            </h1>
            <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
              {lead}
            </p>
            <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-muted-foreground">
              {summary}
            </p>

            <div className="mt-10 flex flex-wrap justify-center gap-4">
              <Button asChild size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 glow-primary px-8">
                <Link to={primaryCta.to}>{primaryCta.label}</Link>
              </Button>
              {secondaryCta ? (
                <Button asChild size="lg" variant="outline" className="border-border text-foreground hover:bg-secondary px-8">
                  <Link to={secondaryCta.to}>{secondaryCta.label}</Link>
                </Button>
              ) : null}
            </div>
          </motion.div>

          <div className="mt-16 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {metrics.map((metric) => (
              <div key={metric.label} className="rounded-2xl border border-white/10 bg-card/80 p-6 shadow-[0_20px_60px_-30px_rgba(0,0,0,0.6)] backdrop-blur">
                <div className="text-3xl font-heading font-bold text-gradient">{metric.value}</div>
                <div className="mt-2 text-sm text-muted-foreground">{metric.label}</div>
              </div>
            ))}
          </div>

          <div className="mt-20 grid gap-6 md:grid-cols-3">
            {highlights.map((highlight) => (
              <motion.div
                key={highlight.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="group rounded-3xl border border-border bg-card p-8 transition duration-300 hover:-translate-y-1 hover:border-primary/30 hover:shadow-[0_24px_70px_-30px_rgba(0,217,255,0.35)]"
              >
                <h3 className="text-xl font-heading font-semibold text-foreground">{highlight.title}</h3>
                <p className="mt-4 text-sm leading-7 text-muted-foreground">{highlight.description}</p>
              </motion.div>
            ))}
          </div>

          <div className="mt-24 space-y-8">
            {sections.map((section, index) => (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className="grid gap-6 rounded-[2rem] border border-white/10 bg-white/5 p-8 backdrop-blur md:grid-cols-[0.8fr_1.2fr]"
              >
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-primary">{section.eyebrow}</div>
                  <h2 className="mt-3 text-2xl font-heading font-bold">{section.title}</h2>
                  <p className="mt-4 text-sm leading-7 text-muted-foreground">{section.description}</p>
                </div>
                <div className="grid gap-3">
                  {section.bullets.map((bullet) => (
                    <div key={bullet} className="rounded-xl border border-border bg-card/70 px-4 py-3 text-sm text-foreground">
                      {bullet}
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </SiteFrame>
  );
};

export type { SectionedPageProps, Metric, Highlight, SectionBlock };
export default SectionedPage;
