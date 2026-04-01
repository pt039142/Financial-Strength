import { Link } from 'react-router-dom';

const footerLinks = {
  Product: [
    { label: "Reconcile AI", href: "/solutions" },
    { label: "Invoice AI", href: "/solutions" },
    { label: "Reporting AI", href: "/products" },
    { label: "API Platform", href: "/developers" },
  ],
  Company: [
    { label: "About", href: "/about" },
    { label: "Careers", href: "/contact" },
    { label: "Blog", href: "/contact" },
    { label: "Press", href: "/contact" },
  ],
  Resources: [
    { label: "Documentation", href: "/developers" },
    { label: "Partners", href: "/contact" },
    { label: "Support", href: "/contact" },
    { label: "Status", href: "/contact" },
  ],
  Legal: [
    { label: "Privacy", href: "/privacy" },
    { label: "Terms", href: "/terms" },
    { label: "Security", href: "/security" },
    { label: "Compliance", href: "/compliance" },
  ],
};

const SiteFooter = () => {
  return (
    <footer className="border-t border-border py-16">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-5">
          <div className="col-span-2 md:col-span-1">
            <Link to="/" className="mb-4 flex items-center gap-2">
              <img
                src="/autonomiq-logo.jpeg"
                alt="Autonomiq.AI"
                className="h-10 w-auto max-w-[190px] rounded-md object-contain"
              />
            </Link>
            <p className="text-sm leading-relaxed text-muted-foreground">
              AI-powered finance automation for the modern enterprise.
            </p>
          </div>

          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="mb-4 font-heading text-sm font-semibold text-foreground">{category}</h4>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.label}>
                    <Link to={link.href} className="text-sm text-muted-foreground transition-colors hover:text-primary">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border pt-8 md:flex-row">
          <p className="text-xs text-muted-foreground">
            © 2026 Autonomiq.AI. All rights reserved.
          </p>
          <div className="flex gap-6">
            {["Twitter", "LinkedIn", "GitHub"].map((social) => (
              <a key={social} href="#" className="text-xs text-muted-foreground transition-colors hover:text-primary">
                {social}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default SiteFooter;
