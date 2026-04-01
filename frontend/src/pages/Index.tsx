import BrandHeroSection from "@/components/BrandHeroSection";
import IntelligenceSection from "@/components/IntelligenceSection";
import StatsSection from "@/components/StatsSection";
import IntegrationsSection from "@/components/IntegrationsSection";
import CTASection from "@/components/CTASection";
import SiteFrame from "@/components/SiteFrame";

const Index = () => {
  return (
    <SiteFrame>
      <BrandHeroSection />
      <StatsSection />
      <IntelligenceSection />
      <IntegrationsSection />
      <CTASection />
    </SiteFrame>
  );
};

export default Index;
