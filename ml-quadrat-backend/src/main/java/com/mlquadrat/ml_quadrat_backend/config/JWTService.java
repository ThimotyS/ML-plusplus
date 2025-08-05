package com.mlquadrat.ml_quadrat_backend.config;

import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;

import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;

// Source: https://github.com/navinreddy20/spring6yt/blob/main/Part38-Spring%20Security%206%20Validating%20JWT%20Token/src/main/java/com/telusko/part29springsecex/service/JWTService.java
@Service
public class JWTService {

    private String secretKey = "";	
	
	public JWTService() {
		
		try {
			KeyGenerator keyGen = KeyGenerator.getInstance("HmacSHA256");
			SecretKey sk = keyGen.generateKey();
			this.secretKey = Base64.getEncoder().encodeToString(sk.getEncoded());
		} catch(NoSuchAlgorithmException e ) {
			throw new RuntimeException(e);
		}
	}
	
	public String generateToken(String name) {
		Map<String, Object> claims = new HashMap<>();
		
		String token = Jwts.builder()
				.claims()
				.add(claims)
				.subject(name)
				.issuedAt(new Date(System.currentTimeMillis()))
				.expiration(new Date(System.currentTimeMillis() + 60 * 60 * 1000000000 ))
				.and().signWith(this.getKey()).compact();
		return token;
	}
	
    private SecretKey getKey() {
        byte[] keyBytes = Decoders.BASE64.decode(this.secretKey);
        return Keys.hmacShaKeyFor(keyBytes);
    }

    public String extractName(String token) {
        return extractClaim(token, Claims::getSubject);
    }

    private <T> T extractClaim(String token, Function<Claims, T> claimResolver) {
        final Claims claims = extractAllClaims(token);
        return claimResolver.apply(claims);
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parser()
                .verifyWith(getKey())
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    public boolean validateToken(String token, UserDetails userDetails) {
        final String userName = extractName(token);
        return (userName.equals(userDetails.getUsername()) && !isTokenExpired(token));
    }

    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }

    private Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }
}
