import { motion } from "framer-motion";

const integrations = [
  "Tally", "Zoho Books", "QuickBooks", "Razorpay", "Stripe", "SAP", "Xero", "FreshBooks"
];

const IntegrationsSection = () => {
  return (
    <section id="platform" className="py-24 relative overflow-hidden">
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="text-xs font-medium text-primary uppercase tracking-widest">Integrations</span>
          <h2 className="text-3xl md:text-4xl font-heading font-bold mt-4 mb-4">
            Connects with Your <span className="text-gradient">Existing Stack</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Seamless integrations with the tools your finance team already uses.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex flex-wrap justify-center gap-4 max-w-3xl mx-auto"
        >
          {integrations.map((name, i) => (
            <motion.div
              key={name}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="px-8 py-4 rounded-xl bg-card border border-border hover:border-primary/30 transition-all duration-300 hover:glow-primary cursor-default"
            >
              <span className="font-heading font-medium text-foreground">{name}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default IntegrationsSection;
