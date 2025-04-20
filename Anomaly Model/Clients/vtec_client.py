import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class VTECClient:
    """Client for retrieving weather alert data from Iowa State University's API."""
    
    BASE_URL = "https://mesonet.agron.iastate.edu/api/1/vtec"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ReTune-Anomaly-Model/1.0",
            "Accept": "application/json"
        })
        
    def get_alerts(self, 
                 begints: Optional[str] = None, 
                 endts: Optional[str] = None, 
                 wfos: Optional[list] = None,
                 only_new: bool = False,
                 include_can: bool = True) -> Dict[str, Any]:
        """
        Fetch weather alerts data from the Mesonet VTEC API.
        
        Parameters:
        -----------
        begints : str, optional
            Start date for data in YYYY-MM-DD format
        endts : str, optional
            End date for data in YYYY-MM-DD format
        wfos : list, optional
            List of Weather Forecast Office codes (e.g., ["OKX", "PHI", "CTP"])
        only_new : bool
            If True, only include new alerts (default: False)
        include_can : bool
            If True, include canceled alerts (default: True)
            
        Returns:
        --------
        Dict[str, Any]
            JSON response containing alert data
        """
        # Set default dates if not provided (last 30 days)
        if not begints:
            begints = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not endts:
            endts = datetime.now().strftime("%Y-%m-%d")
            
        # Build URL parameters
        params = {
            "begints": begints,
            "endts": endts,
            "only_new": str(only_new).lower(),
            "include_can": str(include_can).lower()
        }
        
        # Add WFOs if provided
        if wfos:
            for wfo in wfos:
                params[f"wfo"] = wfo
        
        # Make API request
        url = f"{self.BASE_URL}/sbw_interval.json"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_phenomena_summary(self, 
                            begints: Optional[str] = None, 
                            endts: Optional[str] = None,
                            wfos: Optional[list] = None) -> Dict[str, Any]:
        """
        Get summary of phenomena by type and significance.
        
        Parameters:
        -----------
        begints : str, optional
            Start date for data in YYYY-MM-DD format
        endts : str, optional
            End date for data in YYYY-MM-DD format
        wfos : list, optional
            List of Weather Forecast Office codes (e.g., ["OKX", "PHI", "CTP"])
            
        Returns:
        --------
        Dict[str, Any]
            JSON response containing phenomena summary
        """
        # Set default dates if not provided (last 30 days)
        if not begints:
            begints = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not endts:
            endts = datetime.now().strftime("%Y-%m-%d")
            
        # Build URL parameters
        params = {
            "begints": begints,
            "endts": endts
        }
        
        # Add WFOs if provided
        if wfos:
            for wfo in wfos:
                params[f"wfo"] = wfo
        
        # Make API request
        url = f"{self.BASE_URL}/stats.json"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json() 