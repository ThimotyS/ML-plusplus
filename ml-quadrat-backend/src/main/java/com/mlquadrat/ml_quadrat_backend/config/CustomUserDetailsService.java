package com.mlquadrat.ml_quadrat_backend.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Primary;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import com.mlquadrat.ml_quadrat_backend.mluser.MLUser;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserPrincipal;
import com.mlquadrat.ml_quadrat_backend.mluser.MLUserRepository;

@Service
@Primary
public class CustomUserDetailsService implements UserDetailsService {
	
	@Autowired
	private MLUserRepository repo;
	
	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		MLUser user = repo.findByName(username);
		if (user == null) throw new UsernameNotFoundException("User not found");
		return new MLUserPrincipal(user);
	}
	
}
