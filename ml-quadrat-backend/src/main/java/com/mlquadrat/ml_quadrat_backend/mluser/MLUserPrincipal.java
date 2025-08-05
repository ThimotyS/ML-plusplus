package com.mlquadrat.ml_quadrat_backend.mluser;

import java.util.Collection;

import java.util.Collections;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

public class MLUserPrincipal implements UserDetails {
	
	private MLUser user;
	
	public MLUserPrincipal(MLUser user) {
		this.user = user;
	}
	
	@Override
	public Collection<? extends GrantedAuthority> getAuthorities() {
		return Collections.singleton(new SimpleGrantedAuthority("USER"));
	}

	@Override
	public String getPassword() {
		return user.getPassword();
	}
	
	@Override
	public String getUsername() {
		return user.getName();
	}
	
	public Integer getId() {
		return user.getId();
	}
	
	public MLUser getUser() {
		return user;
	}
	
}
