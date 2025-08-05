package com.mlquadrat.ml_quadrat_backend.mluser;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import com.mlquadrat.ml_quadrat_backend.config.JWTService;

@Service
public class MLUserService {
	
	@Autowired
	MLUserRepository repo;
	
	@Autowired
	AuthenticationManager authenticator;
	
	@Autowired
	private JWTService jwtService;
	
	String verify(MLUser user) {
		Authentication authentication = this.authenticator.authenticate(new UsernamePasswordAuthenticationToken(user.getName(), user.getPassword()));
		if(authentication.isAuthenticated()) return this.jwtService.generateToken(user.getName());
		return "Unsuccessful";
	}

    public MLUser getAuthenticatedUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        MLUserPrincipal userDetails =  (MLUserPrincipal) authentication.getPrincipal();
        return userDetails.getUser();
    }
	
}
